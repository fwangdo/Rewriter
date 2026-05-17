"""ONNX operation type constants used across benchmark models and contracts."""

# Elementwise arithmetic
ADD = "Add"
DIV = "Div"
MUL = "Mul"
NEG = "Neg"
POW = "Pow"
RECIPROCAL = "Reciprocal"
SUB = "Sub"
SUM = "Sum"

# Elementwise math
ABS = "Abs"
ASIN = "Asin"
ATAN = "Atan"
CEIL = "Ceil"
COS = "Cos"
ERF = "Erf"
EXP = "Exp"
FLOOR = "Floor"
LOG = "Log"
MOD = "Mod"
ROUND = "Round"
SIGN = "Sign"
SIN = "Sin"
SQRT = "Sqrt"

# Activation
CLIP = "Clip"
ELU = "Elu"
GELU = "Gelu"
HARD_SIGMOID = "HardSigmoid"
HARD_SWISH = "HardSwish"
LEAKY_RELU = "LeakyRelu"
PRELU = "PRelu"
RELU = "Relu"
SIGMOID = "Sigmoid"
SOFTMAX = "Softmax"
LOG_SOFTMAX = "LogSoftmax"
TANH = "Tanh"
THRESHOLDED_RELU = "ThresholdedRelu"

# Comparison / logic
AND = "And"
EQUAL = "Equal"
GREATER = "Greater"
GREATER_OR_EQUAL = "GreaterOrEqual"
IS_NAN = "IsNaN"
LESS = "Less"
LESS_OR_EQUAL = "LessOrEqual"
NOT = "Not"
OR = "Or"
WHERE = "Where"

# Reduction
REDUCE_L2 = "ReduceL2"
REDUCE_MAX = "ReduceMax"
REDUCE_MEAN = "ReduceMean"
REDUCE_MIN = "ReduceMin"
REDUCE_PROD = "ReduceProd"
REDUCE_SUM = "ReduceSum"

# Linear algebra
GEMM = "Gemm"
MAT_MUL = "MatMul"
EINSUM = "Einsum"

# Convolution / pooling
AVERAGE_POOL = "AveragePool"
CONV = "Conv"
CONV_TRANSPOSE = "ConvTranspose"
GLOBAL_AVERAGE_POOL = "GlobalAveragePool"
GLOBAL_MAX_POOL = "GlobalMaxPool"
MAX_POOL = "MaxPool"

# Normalization
BATCH_NORMALIZATION = "BatchNormalization"
INSTANCE_NORMALIZATION = "InstanceNormalization"
LAYER_NORMALIZATION = "LayerNormalization"
LP_NORMALIZATION = "LpNormalization"

# Shape / layout
CAST = "Cast"
CONCAT = "Concat"
DEPTH_TO_SPACE = "DepthToSpace"
EXPAND = "Expand"
FLATTEN = "Flatten"
IDENTITY = "Identity"
PAD = "Pad"
RESHAPE = "Reshape"
RESIZE = "Resize"
SHAPE = "Shape"
SLICE = "Slice"
SPACE_TO_DEPTH = "SpaceToDepth"
SPLIT = "Split"
SQUEEZE = "Squeeze"
TILE = "Tile"
TRANSPOSE = "Transpose"
UNSQUEEZE = "Unsqueeze"
UPSAMPLE = "Upsample"

# Gather / scatter
GATHER = "Gather"
GATHER_ELEMENTS = "GatherElements"
GATHER_ND = "GatherND"
SCATTER_ELEMENTS = "ScatterElements"
SCATTER_ND = "ScatterND"

# Constant / generation
CONSTANT = "Constant"
CONSTANT_OF_SHAPE = "ConstantOfShape"
RANDOM_UNIFORM_LIKE = "RandomUniformLike"
RANGE = "Range"

# Recurrent
LSTM = "LSTM"

# Quantization
DEQUANTIZE_LINEAR = "DequantizeLinear"
QUANTIZE_LINEAR = "QuantizeLinear"

# Misc
CUM_SUM = "CumSum"
GRID_SAMPLE = "GridSample"
INVERSE = "Inverse"
LRN = "LRN"
MAX = "Max"
MEAN = "Mean"
MIN = "Min"
STFT = "STFT"
TOP_K = "TopK"
TRILU = "Trilu"
