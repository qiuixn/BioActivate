#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import glob
import logging
from multiprocessing import Pool
from rdkit import Chem
from meeko import MoleculePreparation
from vina import Vina
from utils.util import getPath

# 配置日志
logging.basicConfig(
    filename='batch_docking.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BatchDocker:
    def __init__(self, receptor_pdbqt, ligands_dir, output_dir, num_workers=4):
        """
        初始化批量对接器
        :param receptor_pdbqt: 受体PDBQT文件路径
        :param ligands_dir: 配体文件目录（支持.pdb,.sdf）
        :param output_dir: 结果输出目录
        :param num_workers: 并行进程数
        """
        self.receptor = receptor_pdbqt
        self.ligands_dir = ligands_dir
        self.output_dir = output_dir
        self.num_workers = num_workers

        os.makedirs(output_dir, exist_ok=True)

        # 自动检测输入文件类型
        self.ligand_files = glob.glob(os.path.join(ligands_dir, '*.*'))
        logging.info(f"发现{len(self.ligand_files)}个配体文件")

    def _prepare_ligand(self, ligand_file):
        """配体预处理（自动处理多种格式）"""
        try:
            ext = os.path.splitext(ligand_file)[1].lower()

            if ext == '.pdb':
                mol = Chem.MolFromPDBFile(ligand_file, sanitize=False, removeHs=False)
            elif ext == '.sdf':
                mol = Chem.SDMolSupplier(ligand_file)[0]
            else:
                raise ValueError(f"不支持的格式: {ext}")

            # 标准化处理
            mol = Chem.AddHs(mol, addCoords=True)
            preparator = MoleculePreparation()
            preparator.prepare(mol)

            pdbqt_path = os.path.join(
                self.output_dir,
                f"{os.path.basename(ligand_file).split('.')[0]}.pdbqt"
            )
            preparator.write_pdbqt_file(pdbqt_path)
            return pdbqt_path
        except Exception as e:
            logging.error(f"预处理失败: {ligand_file} - {str(e)}")
            return None

    def _dock_single(self, ligand_pdbqt):
        """单配体对接流程"""
        try:
            v = Vina(sf_name='vina', verbosity=0)
            v.set_receptor(self.receptor)
            v.set_ligand_from_file(ligand_pdbqt)

            # 自动计算盒子中心
            box_center = self._detect_binding_site(ligand_pdbqt)
            box_size = [30, 30, 30]  # 可根据需要调整

            v.compute_vina_maps(center=box_center, box_size=box_size)
            v.dock(exhaustiveness=32, n_poses=5)

            # 保存结果
            output_name = os.path.basename(ligand_pdbqt).replace('.pdbqt', '_docked.pdbqt')
            output_path = os.path.join(self.output_dir, output_name)
            v.write_poses(output_path, overwrite=True)

            # 提取最佳亲和力
            best_affinity = v.energies()[0][0]
            return (output_path, best_affinity)
        except Exception as e:
            logging.error(f"对接失败: {ligand_pdbqt} - {str(e)}")
            return None

    def _detect_binding_site(self, ligand_pdbqt):
        """从配体坐标推断结合位点"""
        with open(ligand_pdbqt) as f:
            coords = []
            for line in f:
                if line.startswith('ATOM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
            return [sum(c) / len(c) for c in zip(*coords)]

    def run(self):
        """启动批量对接"""
        with Pool(self.num_workers) as pool:
            # 预处理阶段
            ligands_pdbqt = [
                p for p in pool.map(self._prepare_ligand, self.ligand_files) if p
            ]
            logging.info(f"成功预处理{len(ligands_pdbqt)}个配体")

            # 对接阶段
            results = [
                r for r in pool.map(self._dock_single, ligands_pdbqt) if r
            ]

            # 生成报告
            report = [
                f"配体: {os.path.basename(r[0])}, 亲和力: {r[1]:.2f} kcal/mol"
                for r in results
            ]
            with open(os.path.join(self.output_dir, 'summary.txt'), 'w') as f:
                f.write('\n'.join(report))


if __name__ == "__main__":
    docker = BatchDocker(
        receptor_pdbqt=getPath() + "/data/receptor_clean.pdbqt",
        ligands_dir=getPath() + "/pdbqt_files",
        output_dir=getPath() + "docker_results",
        num_workers=os.cpu_count()  # 自动使用所有CPU核心
    )
    docker.run()