#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import glob
import logging
import warnings
from vina import Vina
from meeko import MoleculePreparation, PDBQTMolecule, prepare_flexreceptor
from rdkit import Chem
from utils.util import getPath

# 配置日志
logging.basicConfig(
    filename='flex_docking.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FlexibleDocker:
    def __init__(self, receptor_pdb, ligands_dir, output_dir, flex_residues):
        """初始化柔性对接参数"""
        # 路径标准化
        self.receptor_pdb = os.path.normpath(os.path.join(getPath(), receptor_pdb))
        self.ligands_dir = os.path.normpath(os.path.join(getPath(), ligands_dir))
        self.output_dir = os.path.normpath(os.path.join(getPath(), output_dir))
        self.flex_residues = flex_residues  # 格式: ["ALA:123", "THR:456"]

        # 准备柔性受体
        self._prepare_flex_receptor()

        # 获取配体文件
        self.ligand_files = [
            f for f in glob.glob(os.path.join(self.ligands_dir, '*'))
            if f.lower().endswith(('.pdbqt', '.sdf', '.mol2'))
        ]
        logging.info(f"Loaded {len(self.ligand_files)} ligand files")

    def _prepare_flex_receptor(self):
        """生成柔性受体文件"""
        try:
            # 生成柔性受体
            self.flex_receptor = prepare_flexreceptor(
                rigid_receptor=self.receptor_pdb,
                flex_residues=self.flex_residues,
                output_root=os.path.join(self.output_dir, "flex_receptors")
            )
            logging.info(f"Flex receptor generated: {self.flex_receptor}")

            # 准备刚性部分参数
            preparer = MoleculePreparation()
            rigid_mol = Chem.MolFromPDBFile(self.receptor_pdb)
            preparer.prepare(rigid_mol)
            self.rigid_pdbqt = os.path.join(self.output_dir, "rigid.pdbqt")
            preparer.write_pdbqt_file(self.rigid_pdbqt)

        except Exception as e:
            logging.error(f"Flex receptor prep failed: {str(e)}")
            raise

    def _prepare_ligand(self, ligand_file):
        """配体预处理"""
        try:
            # 转换不同格式到PDBQT
            if ligand_file.endswith('.pdbqt'):
                return ligand_file

            preparer = MoleculePreparation()
            if ligand_file.endswith('.sdf'):
                mol = Chem.MolFromMolFile(ligand_file)
            elif ligand_file.endswith('.mol2'):
                mol = Chem.MolFromMol2File(ligand_file)

            preparer.prepare(mol)
            output_path = os.path.join(self.output_dir,
                                       os.path.basename(ligand_file).split('.')[0] + ".pdbqt")
            preparer.write_pdbqt_file(output_path)
            return output_path

        except Exception as e:
            logging.error(f"Ligand prep failed: {ligand_file} - {str(e)}")
            return None

    def _dock_single(self, ligand_path):
        """单次柔性对接"""
        try:
            # 加载柔性受体
            v = Vina(verbosity=0, flex=True)
            v.load_receptor(self.rigid_pdbqt, self.flex_receptor)

            # 加载配体
            ligand_pdbqt = self._prepare_ligand(ligand_path)
            if not ligand_pdbqt:
                return None
            v.set_ligand_from_file(ligand_pdbqt)

            # 设置对接参数
            v.compute_vina_maps()
            v.dock(exhaustiveness=128, n_poses=20,
                   min_rmsd=1.5, max_evals=20_000_000)

            # 保存结果
            output_name = os.path.basename(ligand_path).split('.')[0]
            output_path = os.path.join(self.output_dir, f"{output_name}_flex.pdbqt")
            v.write_poses(output_path, overwrite=True, n_poses=20)

            return (output_path, v.energies()[0][0])

        except Exception as e:
            logging.error(f"Docking failed: {ligand_path} - {str(e)}")
            return None

    def run(self):
        """主运行流程"""
        success = 0
        report = ["=== Flexible Docking Report ==="]

        for idx, lig in enumerate(self.ligand_files, 1):
            logging.info(f"Processing {idx}/{len(self.ligand_files)}: {os.path.basename(lig)}")
            result = self._dock_single(lig)

            if result:
                path, energy = result
                report.append(f"{os.path.basename(path)}: {energy:.2f} kcal/mol")
                success += 1

        # 生成报告
        report.extend([
            f"\nSummary:",
            f"Total Ligands: {len(self.ligand_files)}",
            f"Success Rate: {success / len(self.ligand_files):.1%}",
            f"Flex Residues: {', '.join(self.flex_residues)}"
        ])

        with open(os.path.join(self.output_dir, "flex_report.txt"), 'w') as f:
            f.write('\n'.join(report))

        logging.info("Flexible docking completed")


if __name__ == "__main__":
    try:
        # 示例：设置PPARγ的已知柔性残基
        docker = FlexibleDocker(
            receptor_pdbqt="receptor_clean.pdb",
            ligands_dir="pdbqt_files",
            output_dir="flex_results",
            flex_residues=["LEU:330", "PHE:360", "TYR:473"]  # PPARγ关键柔性位点
        )
        docker.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)