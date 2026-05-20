## Goal
- 본 프로젝트의 목표는 하드웨어에서 지원되는 연산의 목록이 있을 때, 모델이 주어질 경우 모델을 수학적으로 동일하면서 지원되는 연산으로만 표현하는 것이 목표. 여기서 핵심은 "변환 규칙"을 새로 찾는 것. 예컨대 conv 는 지원되지만, matmul은 지원되지 않는 경우 conv -> mamtul 변환 규칙을 자동으로 찾는 것. 
- 방법은 e-graph이다. 여기서 자체 ir을 디자인하는데, minimal ir을 정의한 다음, onnx operation 을 분해하고 e-graph로 saturation을 한 다음, sup op로 표현된 onnx로 lifting 되는 후보를 찾는다. 

# Rules
- matmul(no-channel limit.) 
- gemm
- gather
- batch normalization  