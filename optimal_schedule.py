"""
Optimal non-uniform time step schedule for iMF multi-step sampling.

Based on toy Gaussian model analysis: optimizes time points to maximize
variance recovery coefficient c_N for a given NFE and lambda grid.

Usage:
    python optimal_schedule.py              # print default tables
    python optimal_schedule.py --nfe 5      # optimize for specific NFE
"""

import torch


def toy_alpha(t: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    return (((1.0 + lam) * t) - lam) / (t * t + lam * (1.0 - t) * (1.0 - t))


def toy_alpha_prime(t: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    num = lam - (((1.0 + lam) * t) - lam) ** 2
    den = (t * t + lam * (1.0 - t) * (1.0 - t)) ** 2
    return num / den


def toy_cN(times: torch.Tensor, lam: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    h = times[:-1] - times[1:]
    a = toy_alpha(times[:-1], lam)
    factors = 1.0 - h * a
    factors = torch.clamp(factors, min=eps)
    return torch.prod(factors)


def optimize_toy_schedule(
    nfe: int,
    lams=None,
    weights=None,
    max_iter: int = 200,
    lr: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    if lams is None:
        lams = [0.25, 0.5, 1.0, 2.0, 4.0]
    if isinstance(lams, (float, int)):
        lams = [float(lams)]

    lam_t = torch.tensor(lams, dtype=dtype, device=device)

    if weights is None:
        w_t = torch.ones_like(lam_t) / lam_t.numel()
    else:
        w_t = torch.tensor(weights, dtype=dtype, device=device)
        w_t = w_t / w_t.sum()

    logits = torch.nn.Parameter(torch.zeros(nfe, dtype=dtype, device=device))

    opt = torch.optim.LBFGS(
        [logits],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def build_times():
        h = torch.softmax(logits, dim=0)
        times = torch.cat([
            torch.ones(1, dtype=dtype, device=device),
            1.0 - torch.cumsum(h, dim=0),
        ], dim=0)
        times[-1] = torch.tensor(0.0, dtype=dtype, device=device)
        return times, h

    def closure():
        opt.zero_grad()
        times, h = build_times()
        total_loss = torch.zeros((), dtype=dtype, device=device)
        for lam, w in zip(lam_t, w_t):
            a = toy_alpha(times[:-1], lam)
            factors = 1.0 - h * a
            barrier = torch.relu(1e-10 - factors).square().sum() * 1e8
            log_cN = torch.log(torch.clamp(factors, min=1e-12)).sum()
            target = 0.5 * torch.log(lam)
            total_loss = total_loss + w * (log_cN - target).square() + barrier
        total_loss.backward()
        return total_loss

    opt.step(closure)

    with torch.no_grad():
        times, h = build_times()
        cvals = {float(lam): float(toy_cN(times, lam)) for lam in lam_t}
        targets = {float(lam): float(torch.sqrt(lam)) for lam in lam_t}

    return {
        "times": [round(float(x), 6) for x in times.cpu()],
        "step_sizes": [round(float(x), 6) for x in h.cpu()],
        "cN_per_lambda": cvals,
        "target_per_lambda": targets,
    }


def get_default_schedules():
    """Pre-compute optimal schedules for common NFE values."""
    lam_grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    weights = [1, 1, 2, 1, 1]
    schedules = {}
    for nfe in [2, 3, 4, 5, 6, 7, 8, 10]:
        result = optimize_toy_schedule(nfe=nfe, lams=lam_grid, weights=weights)
        schedules[nfe] = result["times"]
    return schedules


# Pre-computed default schedules (run once, paste results)
DEFAULT_SCHEDULES = None


def get_schedule(nfe: int, schedule_type: str = "uniform"):
    """
    Get time step schedule for given NFE.

    Args:
        nfe: number of steps
        schedule_type: 'uniform' or 'optimal'
    Returns:
        list of time points [1.0, ..., 0.0]
    """
    if schedule_type == "uniform":
        return [1.0 - i / nfe for i in range(nfe + 1)]

    elif schedule_type == "optimal":
        global DEFAULT_SCHEDULES
        if DEFAULT_SCHEDULES is None:
            DEFAULT_SCHEDULES = get_default_schedules()

        if nfe in DEFAULT_SCHEDULES:
            return DEFAULT_SCHEDULES[nfe]
        else:
            result = optimize_toy_schedule(nfe=nfe)
            return result["times"]
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfe', type=int, default=None)
    args = parser.parse_args()

    lam_grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    weights = [1, 1, 2, 1, 1]

    if args.nfe:
        nfes = [args.nfe]
    else:
        nfes = [2, 3, 4, 5, 6, 7, 8, 10]

    print("Optimal schedules (lambda grid = [0.25, 0.5, 1.0, 2.0, 4.0]):\n")

    for nfe in nfes:
        result = optimize_toy_schedule(nfe=nfe, lams=lam_grid, weights=weights)

        # Also compute uniform for comparison
        uniform_times = torch.tensor([1.0 - i / nfe for i in range(nfe + 1)], dtype=torch.float64)
        uniform_cN = {float(lam): float(toy_cN(uniform_times, torch.tensor(lam, dtype=torch.float64))) for lam in lam_grid}

        print(f"NFE={nfe}")
        print(f"  Optimal times : {result['times']}")
        print(f"  Optimal steps : {result['step_sizes']}")
        print(f"  c_N (optimal) : { {k: round(v, 4) for k, v in result['cN_per_lambda'].items()} }")
        print(f"  c_N (uniform) : { {k: round(v, 4) for k, v in uniform_cN.items()} }")
        print(f"  c_inf (target): { {k: round(v, 4) for k, v in result['target_per_lambda'].items()} }")
        print()
