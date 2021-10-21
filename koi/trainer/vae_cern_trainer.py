from koi.trainer.vae_trainer import VAETrainer


class VAECernTrainer(VAETrainer):
    def _training_epoch(self, step, epoch, dl):
        device = self.device
        x_dim = self.config.x_dim
        nll, kld, kl_weight = None, None, None

        for iteration, (x, label) in enumerate(dl):
            step += 1
            x, label = x.to(device), label.to(device)
            x = x.reshape(-1, x_dim)
            recon_x, recon_log_var, mean, log_var, z = self.model(x, sample=True)
            kl_weight, beta = self.get_current_kl(step)
            nll, kld, loss = self.model.loss_function(recon_x, x, mean, log_var, kl_weight=kl_weight,
                                                      recon_log_var=recon_log_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # todo tensorboard
        print("nll: {0:.4f}, kld: {1:.4f}, klw: {2:.4f}".format(nll.item(), kld.item(), kl_weight))
        return step
