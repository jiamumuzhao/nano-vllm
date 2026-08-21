import pytest
import torch
import torch.nn.functional as F

from nanovllm.kernels.w8a8_tensorcore import availability, linear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _case(m, n, k, bias=True, implementation="auto"):
    torch.manual_seed(2000 + m + n + k)
    x=torch.randn(m,k,device="cuda",dtype=torch.float16)
    w=torch.randn(n,k,device="cuda",dtype=torch.float16); w[3].zero_()
    ws=w.float().abs().amax(1).clamp_min(1e-12).div(127).half()
    wi=torch.round(w.float()/ws.float()[:,None]).clamp(-127,127).to(torch.int8)
    b=torch.randn(n,device="cuda",dtype=torch.float16) if bias else None
    y=linear(x,wi,ws,b,implementation=implementation)
    # Match the kernel: both activation and weight scales are stored in FP16
    # before quantization/epilogue, so the reference must use FP16-rounded xs.
    xs=x.float().abs().amax(1).clamp_min(1e-12).div(127).half()
    xq=torch.round(x.float()/xs.float()[:,None]).clamp(-127,127)
    ref=((xq@wi.float().t())*xs.float()[:,None]*ws.float()[None,:])
    if b is not None: ref=ref+b.float()
    return y,ref.half()


@pytest.mark.parametrize("m",[1,4,16])
@pytest.mark.parametrize("n",[1024,4096])
def test_w8a8_int8_mma_shapes(m,n):
    y,ref=_case(m,n,1024,True)
    assert linear.last_path=="w8a8_experimental_int8_mma"
    torch.testing.assert_close(y,ref,rtol=.08,atol=.35)


def test_w8a8_non_aligned_3d_no_bias_and_zero_row():
    m,n,k=3,1025,1031
    y,ref=_case(m,n,k,False)
    torch.testing.assert_close(y,ref,rtol=.08,atol=.35)
    x=torch.zeros(4,k,device="cuda",dtype=torch.float16); w=torch.randn(n,k,device="cuda",dtype=torch.float16); w[5].zero_(); ws=w.float().abs().amax(1).clamp_min(1e-12).div(127).half(); wi=torch.round(w.float()/ws.float()[:,None]).clamp(-127,127).to(torch.int8)
    y=linear(x,wi,ws,None)
    assert torch.count_nonzero(y)==0


def test_w8a8_3d_activation_and_non_default_stream():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x=torch.randn(1,4,1024,device="cuda",dtype=torch.float16); w=torch.randn(1024,1024,device="cuda",dtype=torch.float16); ws=w.float().abs().amax(1).clamp_min(1e-12).div(127).half(); wi=torch.round(w.float()/ws.float()[:,None]).clamp(-127,127).to(torch.int8); b=torch.randn(1024,device="cuda",dtype=torch.float16)
        y=linear(x,wi,ws,b); xs=x.float().abs().amax(2).clamp_min(1e-12).div(127); xq=torch.round(x.float()/xs[...,None]).clamp(-127,127); ref=((xq.float()@wi.float().t())*xs[...,None]*ws.float())+b.float(); done=torch.cuda.Event(); done.record(stream)
    done.synchronize()
    assert linear.last_path=="w8a8_experimental_int8_mma"
    torch.testing.assert_close(y,ref.half(),rtol=.08,atol=.35)


@pytest.mark.parametrize("m,n,bias,implementation,expected",[
    (16,1024,True,"cta16x64","w8a8_experimental_int8_mma_cta16x64"),
    (16,4096,False,"cta16x64","w8a8_experimental_int8_mma_cta16x64"),
    (32,1024,True,"cta32x64","w8a8_experimental_int8_mma_cta32x64"),
    (64,4096,False,"cta32x64","w8a8_experimental_int8_mma_cta32x64"),
    (17,1025,True,"cta16x64","w8a8_experimental_int8_mma_cta16x64"),
])
def test_w8a8_cta_target_shapes_and_boundaries(m,n,bias,implementation,expected):
    k=1031 if m==17 else 1024
    y,ref=_case(m,n,k,bias,implementation)
    assert linear.last_path==expected
    torch.testing.assert_close(y,ref,rtol=.08,atol=.40)


def test_w8a8_cta_non_default_stream():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        m,n,k=16,1024,1024; x=torch.randn(m,k,device="cuda",dtype=torch.float16); w=torch.randn(n,k,device="cuda",dtype=torch.float16); ws=w.float().abs().amax(1).clamp_min(1e-12).div(127).half(); wi=torch.round(w.float()/ws.float()[:,None]).clamp(-127,127).to(torch.int8); b=torch.randn(n,device="cuda",dtype=torch.float16); y=linear(x,wi,ws,b,implementation="cta16x64"); xs=x.float().abs().amax(1).clamp_min(1e-12).div(127).half(); xq=torch.round(x.float()/xs.float()[:,None]).clamp(-127,127); ref=((xq@wi.float().t())*xs.float()[:,None]*ws.float()[None,:]+b.float()).half(); done=torch.cuda.Event(); done.record(stream)
    done.synchronize()
    assert linear.last_path=="w8a8_experimental_int8_mma_cta16x64"
    torch.testing.assert_close(y,ref,rtol=.08,atol=.35)


@pytest.mark.parametrize("m,n,bias",[(32,4096,True),(32,4096,False),(64,1024,True),(64,1024,False),(64,4096,True),(64,4096,False)])
def test_w8a8_cta32_cross_grid_correctness(m,n,bias):
    y,ref=_case(m,n,1024,bias,implementation="cta32x64")
    assert linear.last_path=="w8a8_experimental_int8_mma_cta32x64"
    # FP16 scale rounding plus the FP32 accumulator-to-FP16 epilogue can add
    # a sub-half-unit difference on the largest CTA32 grid case.
    torch.testing.assert_close(y,ref,rtol=.08,atol=.40)


def test_w8a8_cta32_non_default_stream():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        m,n,k=64,4096,1024
        x=torch.randn(m,k,device="cuda",dtype=torch.float16)
        w=torch.randn(n,k,device="cuda",dtype=torch.float16)
        ws=w.float().abs().amax(1).clamp_min(1e-12).div(127).half()
        wi=torch.round(w.float()/ws.float()[:,None]).clamp(-127,127).to(torch.int8)
        b=torch.randn(n,device="cuda",dtype=torch.float16)
        y=linear(x,wi,ws,b,implementation="cta32x64")
        xs=x.float().abs().amax(1).clamp_min(1e-12).div(127).half()
        xq=torch.round(x.float()/xs.float()[:,None]).clamp(-127,127)
        ref=((xq@wi.float().t())*xs.float()[:,None]*ws.float()[None,:]+b.float()).half()
        done=torch.cuda.Event(); done.record(stream)
    done.synchronize()
    assert linear.last_path=="w8a8_experimental_int8_mma_cta32x64"
    torch.testing.assert_close(y,ref,rtol=.08,atol=.35)


@pytest.mark.parametrize("m,implementation,expected",[(1,"auto","w8a8_experimental_int8_mma"),(4,"auto","w8a8_experimental_int8_mma"),(17,"auto","w8a8_experimental_int8_mma"),(32,"auto","w8a8_experimental_int8_mma")])
def test_w8a8_auto_is_conservative(m,implementation,expected):
    _case(m,1024,1024,True,implementation)
    assert linear.last_path==expected
