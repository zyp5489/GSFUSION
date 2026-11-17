import numpy as np
import torch
import pytorch_lightning as pl

from ..models.dta_model import DTAProtGraphChemGraph3
from ..metrics          import get_metrics_reg


class testModel(pl.LightningModule):
    def __init__(self, **param_dict):
        super().__init__()
        self.save_hyperparameters(param_dict)
        self.model     = DTAProtGraphChemGraph3(**param_dict)
        self.criterion = param_dict["criterion"]
        self.lr        = param_dict["lr"]
        self.batch_size = param_dict["batch_size"]

        self.L_view_contrastive_weight = param_dict.get("L_view_contrastive_weight", 1.0)
        self.use_cvcl = param_dict.get("use_cvcl", True)

    def forward(self, chem_fp, ligand_graph, protein_graph,
                batch_ligand_token, atom_pad_mask, drug_ids, target_ids, full_aff):
        return self.model(
            chem_fp, ligand_graph, protein_graph,
            batch_ligand_token, atom_pad_mask,
            batch_drug_ids=drug_ids, batch_target_ids=target_ids,
            full_affinity_train=full_aff,
            return_aux_loss=self.use_cvcl)

    def training_step(self, batch, batch_idx):
        full_aff = self.trainer.datamodule.full_affinity.to(self.device)

        (y_true, chem_fp, ligand_graph, protein_graph,
         batch_ligand_token, atom_pad_mask, drug_ids, target_ids) = batch

        with torch.autocast(device_type=self.device.type, enabled=False):
            y_final, y_final2, aux = self.model(
                chem_fp, ligand_graph, protein_graph,
                batch_ligand_token, atom_pad_mask,
                batch_drug_ids=drug_ids, batch_target_ids=target_ids,
                full_affinity_train=full_aff,
                return_aux_loss=self.use_cvcl)

        loss = self.criterion(y_final2.float(), y_true.float())

        if self.use_cvcl:
            loss_cvcl = (aux.get('chem_contrastive', 0.0) +
                         aux.get('prot_contrastive', 0.0) +
                         aux.get('prot_contrastive_2', 0.0) +
                         aux.get('prot_contrastive_3', 0.0))
            loss = loss + loss_cvcl * self.L_view_contrastive_weight

        self.log('train_loss', loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        full_aff = self.trainer.datamodule.full_affinity.to(self.device)

        (y_true, chem_fp, ligand_graph, protein_graph,
         batch_ligand_token, atom_pad_mask, drug_ids, target_ids) = batch

        y_final, y_final2, _ = self.model(
            chem_fp, ligand_graph, protein_graph,
            batch_ligand_token, atom_pad_mask,
            batch_drug_ids=drug_ids, batch_target_ids=target_ids,
            full_affinity_train=full_aff,
            return_aux_loss=False)

        loss_final = self.criterion(y_final.float(), y_true.float())
        pred = (y_final.float() + y_final2.float()) / 2

        np_pred_y = pred.detach().cpu().numpy()
        np_y      = y_true.detach().cpu().numpy()

        metrics = get_metrics_reg(np_y.astype(np.float32),
                                  np_pred_y.astype(np.float32),
                                  "valid", with_rm2=True, with_ci=True)
        metrics["valid_loss"] = loss_final.float()
        self.log_dict(metrics, prog_bar=True, batch_size=self.batch_size)
        return metrics

    def on_train_epoch_start(self):
        torch.cuda.reset_peak_memory_stats()

    def on_validation_epoch_start(self):
        torch.cuda.reset_peak_memory_stats()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr,
                                 weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)