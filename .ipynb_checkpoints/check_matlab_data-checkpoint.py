import scipy.io
import numpy as np
import fitreg
import EQPreg 

def test_with_mat_file(filename):
    print(f"--- Đang kiểm tra file: {filename} ---")
    
    try:
        mat_data = scipy.io.loadmat(filename)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {filename}")
        return

    # Lấy scalar an toàn (ép kiểu float để tránh lỗi numpy scalar)
    def get_scalar(key):
        return float(mat_data[key][0][0])

    fitreg.A = get_scalar('A')
    fitreg.B = get_scalar('B')
    fitreg.Z = get_scalar('Z')
    fitreg.d = get_scalar('d')
    fitreg.pr = get_scalar('pr')
    fitreg.s = get_scalar('s')
    
    print(f"Cấu hình load được: A={fitreg.A}, d={fitreg.d}, Z={fitreg.Z}")

    # Lấy x
    x_sample = mat_data['x'].flatten()
    print(f"Kích thước x: {x_sample.shape}")
    print(f"5 giá trị đầu của x: {x_sample[:5]}") # In thử xem x có bị rỗng hay quá lớn không

    # Lấy fval mẫu
    fval_matlab = get_scalar('fval')
    
    # Chạy tính toán
    try:
        my_fval = fitreg.fitreg(x_sample)
        print(f"MATLAB fval: {fval_matlab}")
        print(f"Python fval: {my_fval}")
        
        diff = abs(fval_matlab - my_fval)
        if diff < 1e-5:
            print("=> THÀNH CÔNG: Kết quả trùng khớp!")
        else:
            print(f"=> CẢNH BÁO: Lệch số liệu (Diff: {diff})")
            
    except Exception as e:
        print(f"LỖI khi chạy fitreg: {e}")
    
    print("-" * 30)

# Chạy test
test_with_mat_file('GA1.mat')
test_with_mat_file('GASQP1.mat')