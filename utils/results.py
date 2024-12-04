import numpy as np

def get_relative_error(K1, K2):
    """Get the relative error between two camera matrices. Trim small values to avoid division by zero"""
    diff = np.abs(K1 - K2)
    diff[diff < 1e-10] = 0

    dividend = K1
    dividend[np.abs(dividend) < 1e-10] = 0
    dividend[dividend == 0] = 1

    return diff / dividend

def display_relative_error(err):
    fx_err = err[0, 0]
    fy_err = err[1, 1]
    skew_err = err[0, 1]
    u_err = err[0, 2]
    v_err = err[1, 2]

    print(f"fx error: {fx_err}")
    print(f"fy error: {fy_err}")
    print(f"skew error: {skew_err}")
    print(f"u error: {u_err}")
    print(f"v error: {v_err}")

def write_results(results, filename):
    buf = "fx,fy,skew,u,v\n"
    for res in results:
        fx = res[0, 0]
        fy = res[1, 1]
        skew = res[0, 1]
        u = res[0, 2]
        v = res[1, 2]

        buf += f"{fx},{fy},{skew},{u},{v}\n"
    
    with open(filename, 'w') as f:
        f.write(buf)

def write_errors(errors, filename):
    buf = "fx(rel),fy(rel),skew,u,v\n"
    for err in errors:
        fx_err = err[0, 0]
        fy_err = err[1, 1]
        skew_err = err[0, 1]
        u_err = err[0, 2]
        v_err = err[1, 2]

        buf += f"{fx_err},{fy_err},{skew_err},{u_err},{v_err}\n"
    
    with open(filename, 'w') as f:
        f.write(buf)