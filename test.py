import core

t = core.as_tensor(1)

print(core.ones_like(t))
t.to('gpu')
t.to('cuda')
print(t.data.device)