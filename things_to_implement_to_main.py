    def estimate_sqnr(self, t, sparsity):
        ste = TopKMaskStraightThrough()
        t_s = ste.forward(None, torch.abs(t), sparsity) * t
        mse = torch.mean((t - t_s) ** 2)
        tensor_norm = torch.mean(t**2)
        if mse.item() > 0.0:
            pruning_sqnr = 10 * np.log10(tensor_norm.item() / mse.item())
        else:
            pruning_sqnr = np.Inf
        return (mse, pruning_sqnr)
