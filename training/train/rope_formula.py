import math


def cal_theta(target_context_length, context_pretrain=4096, theta_pretrain=10000):
    theta_new = theta_pretrain ** (
        math.log(target_context_length / (2 * math.pi))
        / math.log(context_pretrain / (2 * math.pi))
    )
    return theta_new


context_lengthes = [4096, 32768, 32768 * 2]
thetas = []
for c in context_lengthes:
    print(f"Context length {c}: suggested theta {cal_theta(c):.2f}")
    thetas.append(cal_theta(c))
print(thetas)
