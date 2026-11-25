
from quantization.quantization_tables import (
    QCL,
    QCR,
    scale_quantization_table
)

qcl_low = scale_quantization_table(QCL, 25)
qcl_high = scale_quantization_table(QCL, 75)

print("Original QCL[0][0]:", QCL[0][0])
print("QCL at quality 25:", qcl_low[0][0])
print("QCL at quality 75:", qcl_high[0][0])

print("\nQCL at quality 25:")
for row in qcl_low:
    print(row)

print("\nQCL at quality 75:")
for row in qcl_high:
    print(row)

qcr_high = scale_quantization_table(QCR, 75)

print("\nQCR at quality 75:")
for row in qcr_high:
    print(row)

qcr_low = scale_quantization_table(QCR, 25)

print("\nQCR at quality 25:")
for row in qcr_low:
    print(row)
