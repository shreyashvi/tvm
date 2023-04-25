#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>

#include "../../utils.h"


#include "../../../../runtime/contrib/json/json_node.h"
#include "../codegen_json/codegen_json.h"

#include "../codegen_c/codegen_c.h"

using namespace std;
namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;


static tvm::Array<Expr> BindToCallNodeArgs(const vector<Expr>& args, const CallNode* cn) {
  tvm::Array<Expr> res;
  for (const auto& arg : args) {
    if (arg->IsInstance<ConstantNode>()) {
      res.push_back(arg);
    } else {
      auto body_params = cn->op.as<FunctionNode>()->params;
      auto found = find(body_params.begin(), body_params.end(), arg);
      ICHECK(found != body_params.end());
      auto idx = distance(body_params.begin(), found);
      res.push_back(cn->args[idx]);
    }
  }
  return res;
}
 // C source runtime
 
inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

inline std::string GetShapeString(std::vector<int> shape) {
  std::string v = "std::vector<long int>{";
  for (auto s : shape) {
    v += std::to_string(s) + ",";
  }
  v += "}";
  return v;
}

std::vector<std::string> Conv2d(const CallNode* call) {
  std::vector<std::string> args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  ICHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: O, G, Ph0, Pw0, Ph1, Pw1, Kh, Kw, Sh, Sw
  args.push_back(std::to_string(wshape[0]));
  args.push_back(std::to_string(conv2d_attr->groups));
  args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->padding[2].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->padding[3].as<IntImmNode>()->value));
  args.push_back(std::to_string(wshape[2]));
  args.push_back(std::to_string(wshape[3]));
  args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

  return args;
}

vector<string> Dense(const CallNode* call) {
  vector<string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  // Args: N, C, O
  args.push_back(to_string(ishape[0]));
  args.push_back(to_string(ishape[1]));
  args.push_back(to_string(wshape[0]));

  return args;
}

vector<string> Relu(const CallNode* call) {
  vector<string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  // Args: N, C, H, W
  args.push_back(GetShapeString(ishape));
  return args;
}
vector<string> BatchNorm(const CallNode* call) {
  vector<string> args;
  const auto* bn_attr = call->attrs.as<BatchNormAttrs>();
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(to_string(s));
  }

  // Args: epsilon
  args.push_back(to_string(bn_attr->epsilon));

  return args;
}

vector<string> Add(const CallNode* call) {
  vector<string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  args.push_back(to_string(0));
  // Args: H, W
  args.push_back(GetShapeString(ishape));
  return args;
}
/*
vector<std::string> Add(const CallNode* call) {
  vector<string> args;
  
  return args;
}
*/
vector<string> Multiply(const CallNode* call) {
  vector<string> args;
  
  
  return args;
}


//create a class CodegenShakti extends with CodegenCBase and MemoizedExprTranslator<:vector<Output>>
class CodegenShakti:public CodegenCBase,public MemoizedExprTranslator<vector<Output>>{
  //constructor
  public:
    explicit CodegenShakti(const string& id){
      this->ext_func_id_=id;
    }
    
    //default when shakti codegen not support the operator
    vector<Output> VisitExprDefault_(const Object *op)final{
      LOG(FATAL)<<"Shakti codegen doesnot support " ;//<<op->GetTypeKey;
      return {};
    }

  //different type of nodes 
  //VisitExpr and VisitExpr_are 2 methods, relation between them is related to dispatch
    vector<Output> VisitExpr_(const VarNode *node)final{
      ext_func_args_.push_back(GetRef<Var>(node));
      Output output;
      output.name=node->name_hint();
      return {output};
    }

    //expression is a tuple node with constants

    vector<Output> VisitExpr_(const TupleNode * node)final{
      vector<Output>outs;
      for(auto x :node->fields)
      {
        auto res=VisitExpr(x);
        CHECK_EQ(res.size(),1U)<<"do not support tuple nest";
        outs.push_back(res[0]);
      }
      return outs;
    }

    vector<Output> VisitExpr_(const TupleGetItemNode *node)final{
      auto res=VisitExpr(node->tuple);
      CHECK_GT(res.size(),static_cast<size_t>(node->index));

      return {res[node->index]};
    }
    vector<Output> VisitExpr_(const ConstantNode* cn) final {
    Output output;
    // Get const: static_cast<float*>(shakti_0_consts[0]->data)
    output.name = CreateDataReference(ext_func_id_, const_idx_);
    output.dtype = "float";

    // Generate the global variable for needed ndarrays
    if (const_array_name_.empty()) {
      const_array_name_ = CreateNDArrayPool(ext_func_id_);
     string checker = CreateInitChecker(ext_func_id_);
      ext_func_body_.insert(ext_func_body_.begin(), checker);
    }

    // Give the ndarray a unique name to ease the initialization of it at
    // runtime.
    string const_var_name = CreateConstVar(ext_func_id_, const_idx_);
    const_vars_.push_back(const_var_name);
    const_idx_++;

    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    ICHECK_EQ(GetDtypeString(type_node), "float") << "Only float is supported for now.";

    return {output};
  }
    vector<Output> VisitExpr_(const CallNode *node)final{
      GenerateBodyOutput ret;
      if(const auto * func=node->op.as<FunctionNode>())
      {
        //ret=GenerateCompositeFunctionCall(func,node);
      }
      else
      ret=GenerateOpCall(node);
		//LOG(INFO)<<"SHREYA";
      buf_decl_.insert(buf_decl_.end(),ret.buffers.begin(),ret.buffers.end());
      ext_func_body_.push_back(ret.decl);
      //LOG(INFO)<<"SHREYA";
      return ret.outputs;

    }
    string JIT(const vector<Output>&out){
      return JitImpl(ext_func_id_,ext_func_args_,buf_decl_,ext_func_body_,const_array_name_,out);
    }




  private:
  //structure
    struct GenerateBodyOutput{
        string decl;
        vector<string>buffers;
        vector<Output>outputs;
    };

    vector<string> GetArgumentNames(const CallNode *node)
    {	
    	//LOG(INFO)<<node->args.size();
      vector<string>ret;
      for(size_t i=0;i<node->args.size();i++)
      {	
      		//LOG(INFO)<<"123";
        auto res=VisitExpr(node->args[i]);
       // LOG(INFO)<<"123";
        for(const auto &out:res)
        {
          ret.push_back(out.name);
        }
       // LOG(INFO)<<"123";
      }
      return ret;
    }

    //GenerateOpcall when single function node

    GenerateBodyOutput GenerateOpCall(const CallNode *node){
      const auto *op_node=node->op.as<OpNode>();
      CHECK(op_node)<<"expect op node but got "<<node->op->GetTypeKey();
      using ArgFunType=function<vector<string>(const CallNode *)>;

      //mapping which functions are supported by shakti
      static const map<string, pair<string, ArgFunType>> op_map = {
        {"nn.conv2d", {"shakti_conv2d", Conv2d}}, {"nn.dense", {"shakti_dense", Dense}},
        {"nn.relu", {"shakti_relu", Relu}},       {"nn.batch_norm", {"shakti_bn", BatchNorm}},
        {"add", {"shakti_binary_op", Add}},       {"multiply", {"shakti_binary_op", Multiply}},
    };

        //get the operator name
      const auto op_name=GetRef<Op>(op_node)->name;
      //check if it is in map or not
      const auto iter=op_map.find(op_name);
      if(iter!=op_map.end())
      { 
      	//LOG(INFO)<<"SHREYA";
        //if found call generate body by passing call node , fucntion naem and
        return GenerateBody(node,iter->second.first,iter->second.second(node));
      }
      LOG(FATAL)<<"unsupported op by shakti"<< AsText(node->op,false);
      //return {};



    }
    // GenerateBodyOutput GenerateBody(const CallNode *node, string &func_name, vector<string>& attribute_args)
    // {
    //   return GenerateBody(node,func_name,GetArgumentNames(node),attribute_args);
    // }

    GenerateBodyOutput GenerateBody(const CallNode* root_call, const string& func_name,
                                  const vector<string>& attribute_args) {
                                  //LOG(INFO)<<"SHREYA";
    return GenerateBody(root_call, func_name, GetArgumentNames(root_call), attribute_args);
  }

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const string& func_name,
                                  const vector<string>& func_args,
                                  const vector<string>& attribute_args) {
    // Make function call with input buffers when visiting arguments
   // LOG(INFO)<<"SHREYA";
    ICHECK_GT(func_args.size(), 0);
    ostringstream decl_stream;
    decl_stream << "(" << func_args[0];
    for (size_t i = 1; i < func_args.size(); ++i) {
      decl_stream << ", " << func_args[i];
    }

    // Analyze the output buffers
    vector<Type> out_types;
    if (root_call->checked_type()->IsInstance<TupleTypeNode>()) {
      auto type_node = root_call->checked_type().as<TupleTypeNode>();
      for (auto field : type_node->fields) {
        ICHECK(field->IsInstance<TensorTypeNode>());
        out_types.push_back(field);
      }
    } else if (root_call->checked_type()->IsInstance<TensorTypeNode>()) {
      ICHECK(root_call->checked_type()->IsInstance<TensorTypeNode>());
      out_types.push_back(root_call->checked_type());
    } else {
      LOG(FATAL) << "Unrecognized type node: " << AsText(root_call->checked_type(), false);
    }

    GenerateBodyOutput ret;
    for (const auto& out_type : out_types) {
      this->PrintIndents();
      const string out = "buf_" + to_string(buf_idx_++);
      const auto out_size = GetShape1DSize(out_type);
      decl_stream << ", " << out;

      Output output;
      output.name = out;
      output.size = out_size;
      output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
      output.need_copy = true;
      ret.buffers.push_back("float* " + out + " = (float*)std::malloc(4 * " +
                            to_string(out_size) + ");");
                           // LOG(INFO)<<"SHREYA";
      ret.outputs.push_back(output);
    }

    // Attach attribute arguments
    for (size_t i = 0; i < attribute_args.size(); ++i) {
      decl_stream << ", " << attribute_args[i];
    }
    decl_stream << ");";
    ret.decl = func_name + decl_stream.str();
    return ret;
  }






    string ext_func_id_{""};// id of externel shakti fucntion
    int buf_idx_{0};
    int const_idx_{0};
    //arguments used by wrapped functions which call shakti kernels
    Array<Var> ext_func_args_;
    //fucntion statements which will be compiled by shakti kernel
    vector<string> ext_func_body_;
    //array to store constant values
    string const_array_name_;
    //immediate buffers
    vector<string> buf_decl_;
    //variable name to constant mapping 
    Array<String> const_vars_;

    friend class ShaktiModuleCodegen;


};

class ShaktiModuleCodegen: public CSourceModuleCodegenBase{
  public:
    //generate Shaktimaan  C function for the relay function
    pair<string, Array<String>> GenShaktiFunc(const Function& func){
      //check is the function is relay function or not
      CHECK(func.defined())<<"Given function is not a relay function";

      auto sid=GetExtSymbol(func);

      CodegenShakti builder(sid);
     // LOG(INFO)<<"SHREYA";
      auto out=builder.VisitExpr(func->body);
      //LOG(INFO)<<"SHREYA";
      code_stream_<<builder.JIT(out);
	//LOG(INFO)<<"SHREYA";
      return {sid,builder.const_vars_};

    }
    //fucntion to generate CSourceModule
    //argument as relay function or a module
    //return runtime module that constains C source code

    runtime::Module CreateCSourceModule(const ObjectRef& ref)override{
      // Create headers
            code_stream_ << "#include <cstdint>\n";
            code_stream_ << "#include <cstdlib>\n";
            code_stream_ << "#include <cstring>\n";
            code_stream_ << "#include <vector>\n";
            code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
            code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
            code_stream_ << "#include <dlpack/dlpack.h>\n";
            // shakti_kernel file is saved under src/runtime/contrib/shakti so that we don't
            // expose it to ordinary users. To make export_library use it, users need to
            // pass -I${PATH_TO_TVM}/src/runtime/contrib
            code_stream_ << "#include <shakti/shakti_kernel.h>\n";
            code_stream_ << "using namespace tvm::runtime;\n";
            code_stream_ << "using namespace tvm::runtime::contrib;\n";
            code_stream_ << "\n";

            // check if argument is an instance of function node or not
            CHECK(ref->IsInstance<FunctionNode>());

            // call GenShaktiFun and add to code_stream
            auto res=GenShaktiFunc(Downcast<Function>(ref));
            string code=code_stream_.str();
            String sym=get<0>(res);
            Array<String> variables=get<1>(res);
		//LOG(INFO)<<"SHREYA";
            //create a Csource module

            //the fucntions returns PackedFunction pointer to register function and nullptr if not exist
            const auto *pf=runtime::Registry::Get("runtime.CSourceModuleCreate");
            //LOG(INFO)<<"SHREYA";
            CHECK(pf!=nullptr)<<"No csource module to create external runtime module";
            return (*pf)(code,"c",Array<String>{sym},variables);




    }
    private:
      ostringstream code_stream_;
    

};
string GetExtSymbol(const Function &func){
  const auto name_node=func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  CHECK(name_node.defined())<<"Fail to retrive external symbol";
  return string(name_node.value());

}



//Register a codegen
//Function to invoke shakticodegen and generate runtime  module

runtime::Module ShaktiCompiler(const ObjectRef& ref) {
  ShaktiModuleCodegen shakti;
  return shakti.CreateCSourceModule(ref);
}
//Register the above ShaktiCompiler function to TVM backend
TVM_REGISTER_GLOBAL("relay.ext.shakti").set_body_typed(ShaktiCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
