# -*- coding: utf-8 -*-
"""漂绿得分 (Greenwashing Score, GWS) 构造.

GWS = z(ESG_dis) - z(ESG_per), 在给定分组内做截面标准化:
  披露得分显著高于同组实际表现 => 漂绿倾向高.
方法上等价于量化因子研究中的"行业中性化 z-score".

注: 原 Stata 代码的行业×年度口径存在 bug(标准差只按行业分组),
本模块 add_gws() 全部采用正确口径; replicate_stata_bug() 保留错误
口径仅用于与历史结果对拍.
"""
import pandas as pd


def zscore(df, col, group):
    g = df.groupby(group, observed=True)[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def zscore_gap(df, group, dis="ESG_dis", per="ESG_per"):
    """组内 z 分数之差: z(披露|group) - z(表现|group)"""
    return zscore(df, dis, group) - zscore(df, per, group)


def add_gws(df):
    """三种口径的 GWS(均为正确口径)."""
    df = df.copy()
    df["GWS_ind"] = zscore_gap(df, "industry")
    df["GWS_indyr"] = zscore_gap(df, ["industry", "year"])
    df["GWS_yr"] = zscore_gap(df, "year")
    return df


def replicate_stata_bug(df):
    """复现原 Stata 'GWS_industryyear': 均值按行业×年, 标准差只按行业(bug)."""
    g_iy = df.groupby(["industry", "year"], observed=True)
    g_i = df.groupby("industry", observed=True)
    return ((df["ESG_dis"] - g_iy["ESG_dis"].transform("mean"))
            / g_i["ESG_dis"].transform("std")
            - (df["ESG_per"] - g_iy["ESG_per"].transform("mean"))
            / g_i["ESG_per"].transform("std"))
