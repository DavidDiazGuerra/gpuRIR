import numpy as np
import matplotlib.pyplot as plt

omni =      [0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246]
homni =     [0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246,0.013246]
card =      [0.013238,0.013237,0.013234,0.013229,0.013222,0.013213,0.013202,0.013189,0.013174,0.013157,0.013138,0.013117,0.013094,0.013069,0.013042,0.013013,0.012982,0.012949,0.012914,0.012878,0.012839,0.012799,0.012756,0.012712,0.012666,0.012618,0.012569,0.012517,0.012464,0.012409,0.012352,0.012293,0.012233,0.012171,0.012107,0.012042,0.011975,0.011906,0.011836,0.011764,0.011691,0.011616,0.011539,0.011461,0.011382,0.011301,0.011218,0.011135,0.011049,0.010963,0.010875,0.010786,0.010696,0.010604,0.010511,0.010417,0.010322,0.010226,0.010129,0.010030,0.009931,0.009830,0.009729,0.009626,0.009523,0.009419,0.009314,0.009208,0.009101,0.008994,0.008886,0.008777,0.008667,0.008557,0.008447,0.008335,0.008223,0.008111,0.007999,0.007885,0.007772,0.007658,0.007544,0.007429,0.007315,0.007200,0.007085,0.006969,0.006854,0.006739,0.006623,0.006508,0.006392,0.006277,0.006162,0.006047,0.005932,0.005817,0.005703,0.005588,0.005475,0.005361,0.005248,0.005135,0.005023,0.004911,0.004800,0.004689,0.004579,0.004470,0.004361,0.004253,0.004145,0.004039,0.003933,0.003828,0.003723,0.003620,0.003518,0.003416,0.003316,0.003216,0.003118,0.003020,0.002924,0.002829,0.002735,0.002642,0.002551,0.002460,0.002371,0.002283,0.002197,0.002112,0.002028,0.001946,0.001865,0.001785,0.001707,0.001631,0.001556,0.001482,0.001411,0.001340,0.001272,0.001205,0.001139,0.001075,0.001013,0.000953,0.000894,0.000838,0.000783,0.000729,0.000678,0.000628,0.000580,0.000534,0.000490,0.000448,0.000407,0.000369,0.000332,0.000297,0.000265,0.000234,0.000205,0.000178,0.000153,0.000130,0.000109,0.000090,0.000073,0.000058,0.000045,0.000033,0.000024,0.000017,0.000012,0.000009,0.000008,0.000009,0.000012,0.000017,0.000024,0.000033,0.000045,0.000058,0.000073,0.000090,0.000109,0.000130,0.000153,0.000178,0.000205,0.000234,0.000265,0.000297,0.000332,0.000369,0.000407,0.000448,0.000490,0.000534,0.000580,0.000628,0.000678,0.000729,0.000783,0.000838,0.000894,0.000953,0.001013,0.001075,0.001139,0.001205,0.001272,0.001340,0.001411,0.001482,0.001556,0.001631,0.001707,0.001785,0.001865,0.001946,0.002028,0.002112,0.002197,0.002283,0.002371,0.002460,0.002551,0.002642,0.002735,0.002829,0.002924,0.003020,0.003118,0.003216,0.003316,0.003416,0.003518,0.003620,0.003723,0.003828,0.003933,0.004039,0.004145,0.004253,0.004361,0.004470,0.004579,0.004689,0.004800,0.004911,0.005023,0.005135,0.005248,0.005361,0.005475,0.005588,0.005703,0.005817,0.005932,0.006047,0.006162,0.006277,0.006392,0.006508,0.006623,0.006739,0.006854,0.006969,0.007085,0.007200,0.007315,0.007429,0.007544,0.007658,0.007772,0.007885,0.007999,0.008111,0.008223,0.008335,0.008447,0.008557,0.008667,0.008777,0.008886,0.008994,0.009101,0.009208,0.009314,0.009419,0.009523,0.009626,0.009729,0.009830,0.009931,0.010030,0.010129,0.010226,0.010322,0.010417,0.010511,0.010604,0.010696,0.010786,0.010875,0.010963,0.011049,0.011135,0.011218,0.011301,0.011382,0.011461,0.011539,0.011616,0.011691,0.011764,0.011836,0.011906,0.011975,0.012042,0.012107,0.012171,0.012233,0.012293,0.012352,0.012409,0.012464,0.012517,0.012569,0.012618,0.012666,0.012712,0.012756,0.012799,0.012839,0.012878,0.012914,0.012949,0.012982,0.013013,0.013042,0.013069,0.013094,0.013117,0.013138,0.013157,0.013174,0.013189,0.013202,0.013213,0.013222,0.013229,0.013234,0.013237]
hypcard =   [0.013238,0.013237,0.013234,0.013229,0.013222,0.013213,0.013202,0.013189,0.013174,0.013157,0.013138,0.013117,0.013094,0.013069,0.013042,0.013013,0.012982,0.012949,0.012914,0.012878,0.012839,0.012799,0.012756,0.012712,0.012666,0.012618,0.012569,0.012517,0.012464,0.012409,0.012352,0.012293,0.012233,0.012171,0.012107,0.012042,0.011975,0.011906,0.011836,0.011764,0.011691,0.011616,0.011539,0.011461,0.011382,0.011301,0.011218,0.011135,0.011049,0.010963,0.010875,0.010786,0.010696,0.010604,0.010511,0.010417,0.010322,0.010226,0.010129,0.010030,0.009931,0.009830,0.009729,0.009626,0.009523,0.009419,0.009314,0.009208,0.009101,0.008994,0.008886,0.008777,0.008667,0.008557,0.008447,0.008335,0.008223,0.008111,0.007999,0.007885,0.007772,0.007658,0.007544,0.007429,0.007315,0.007200,0.007085,0.006969,0.006854,0.006739,0.006623,0.006508,0.006392,0.006277,0.006162,0.006047,0.005932,0.005817,0.005703,0.005588,0.005475,0.005361,0.005248,0.005135,0.005023,0.004911,0.004800,0.004689,0.004579,0.004470,0.004361,0.004253,0.004145,0.004039,0.003933,0.003828,0.003723,0.003620,0.003518,0.003416,0.003316,0.003216,0.003118,0.003020,0.002924,0.002829,0.002735,0.002642,0.002551,0.002460,0.002371,0.002283,0.002197,0.002112,0.002028,0.001946,0.001865,0.001785,0.001707,0.001631,0.001556,0.001482,0.001411,0.001340,0.001272,0.001205,0.001139,0.001075,0.001013,0.000953,0.000894,0.000838,0.000783,0.000729,0.000678,0.000628,0.000580,0.000534,0.000490,0.000448,0.000407,0.000369,0.000332,0.000297,0.000265,0.000234,0.000205,0.000178,0.000153,0.000130,0.000109,0.000090,0.000073,0.000058,0.000045,0.000033,0.000024,0.000017,0.000012,0.000009,0.000008,0.000009,0.000012,0.000017,0.000024,0.000033,0.000045,0.000058,0.000073,0.000090,0.000109,0.000130,0.000153,0.000178,0.000205,0.000234,0.000265,0.000297,0.000332,0.000369,0.000407,0.000448,0.000490,0.000534,0.000580,0.000628,0.000678,0.000729,0.000783,0.000838,0.000894,0.000953,0.001013,0.001075,0.001139,0.001205,0.001272,0.001340,0.001411,0.001482,0.001556,0.001631,0.001707,0.001785,0.001865,0.001946,0.002028,0.002112,0.002197,0.002283,0.002371,0.002460,0.002551,0.002642,0.002735,0.002829,0.002924,0.003020,0.003118,0.003216,0.003316,0.003416,0.003518,0.003620,0.003723,0.003828,0.003933,0.004039,0.004145,0.004253,0.004361,0.004470,0.004579,0.004689,0.004800,0.004911,0.005023,0.005135,0.005248,0.005361,0.005475,0.005588,0.005703,0.005817,0.005932,0.006047,0.006162,0.006277,0.006392,0.006508,0.006623,0.006739,0.006854,0.006969,0.007085,0.007200,0.007315,0.007429,0.007544,0.007658,0.007772,0.007885,0.007999,0.008111,0.008223,0.008335,0.008447,0.008557,0.008667,0.008777,0.008886,0.008994,0.009101,0.009208,0.009314,0.009419,0.009523,0.009626,0.009729,0.009830,0.009931,0.010030,0.010129,0.010226,0.010322,0.010417,0.010511,0.010604,0.010696,0.010786,0.010875,0.010963,0.011049,0.011135,0.011218,0.011301,0.011382,0.011461,0.011539,0.011616,0.011691,0.011764,0.011836,0.011906,0.011975,0.012042,0.012107,0.012171,0.012233,0.012293,0.012352,0.012409,0.012464,0.012517,0.012569,0.012618,0.012666,0.012712,0.012756,0.012799,0.012839,0.012878,0.012914,0.012949,0.012982,0.013013,0.013042,0.013069,0.013094,0.013117,0.013138,0.013157,0.013174,0.013189,0.013202,0.013213,0.013222,0.013229,0.013234,0.013237]
subcard =   [0.013242,0.013242,0.013240,0.013238,0.013234,0.013230,0.013224,0.013218,0.013210,0.013202,0.013192,0.013181,0.013170,0.013157,0.013144,0.013130,0.013114,0.013098,0.013080,0.013062,0.013043,0.013023,0.013001,0.012979,0.012956,0.012932,0.012907,0.012882,0.012855,0.012828,0.012799,0.012770,0.012740,0.012709,0.012677,0.012644,0.012611,0.012576,0.012541,0.012505,0.012468,0.012431,0.012393,0.012354,0.012314,0.012273,0.012232,0.012190,0.012148,0.012105,0.012061,0.012016,0.011971,0.011925,0.011879,0.011832,0.011784,0.011736,0.011687,0.011638,0.011589,0.011538,0.011488,0.011436,0.011385,0.011333,0.011280,0.011227,0.011174,0.011120,0.011066,0.011012,0.010957,0.010902,0.010846,0.010791,0.010735,0.010679,0.010622,0.010566,0.010509,0.010452,0.010395,0.010338,0.010280,0.010223,0.010165,0.010108,0.010050,0.009992,0.009935,0.009877,0.009819,0.009762,0.009704,0.009647,0.009589,0.009532,0.009474,0.009417,0.009360,0.009304,0.009247,0.009191,0.009135,0.009079,0.009023,0.008968,0.008913,0.008858,0.008804,0.008749,0.008696,0.008642,0.008590,0.008537,0.008485,0.008433,0.008382,0.008331,0.008281,0.008231,0.008182,0.008133,0.008085,0.008038,0.007991,0.007944,0.007898,0.007853,0.007809,0.007765,0.007722,0.007679,0.007637,0.007596,0.007556,0.007516,0.007477,0.007439,0.007401,0.007364,0.007328,0.007293,0.007259,0.007225,0.007193,0.007161,0.007130,0.007100,0.007070,0.007042,0.007014,0.006988,0.006962,0.006937,0.006913,0.006890,0.006868,0.006847,0.006827,0.006808,0.006789,0.006772,0.006755,0.006740,0.006726,0.006712,0.006700,0.006688,0.006678,0.006668,0.006660,0.006652,0.006645,0.006640,0.006635,0.006632,0.006629,0.006628,0.006627,0.006628,0.006629,0.006632,0.006635,0.006640,0.006645,0.006652,0.006660,0.006668,0.006678,0.006688,0.006700,0.006712,0.006726,0.006740,0.006755,0.006772,0.006789,0.006808,0.006827,0.006847,0.006868,0.006890,0.006913,0.006937,0.006962,0.006988,0.007014,0.007042,0.007070,0.007100,0.007130,0.007161,0.007193,0.007225,0.007259,0.007293,0.007328,0.007364,0.007401,0.007439,0.007477,0.007516,0.007556,0.007596,0.007637,0.007679,0.007722,0.007765,0.007809,0.007853,0.007898,0.007944,0.007991,0.008038,0.008085,0.008133,0.008182,0.008231,0.008281,0.008331,0.008382,0.008433,0.008485,0.008537,0.008590,0.008642,0.008696,0.008749,0.008804,0.008858,0.008913,0.008968,0.009023,0.009079,0.009135,0.009191,0.009247,0.009304,0.009360,0.009417,0.009474,0.009532,0.009589,0.009647,0.009704,0.009762,0.009819,0.009877,0.009935,0.009992,0.010050,0.010108,0.010165,0.010223,0.010280,0.010338,0.010395,0.010452,0.010509,0.010566,0.010622,0.010679,0.010735,0.010791,0.010846,0.010902,0.010957,0.011012,0.011066,0.011120,0.011174,0.011227,0.011280,0.011333,0.011385,0.011436,0.011488,0.011538,0.011589,0.011638,0.011687,0.011736,0.011784,0.011832,0.011879,0.011925,0.011971,0.012016,0.012061,0.012105,0.012148,0.012190,0.012232,0.012273,0.012314,0.012354,0.012393,0.012431,0.012468,0.012505,0.012541,0.012576,0.012611,0.012644,0.012677,0.012709,0.012740,0.012770,0.012799,0.012828,0.012855,0.012882,0.012907,0.012932,0.012956,0.012979,0.013001,0.013023,0.013043,0.013062,0.013080,0.013098,0.013114,0.013130,0.013144,0.013157,0.013170,0.013181,0.013192,0.013202,0.013210,0.013218,0.013224,0.013230,0.013234,0.013238,0.013240,0.013242]
bidir=      [0.013230,0.013228,0.013222,0.013212,0.013198,0.013179,0.013157,0.013131,0.013101,0.013067,0.013029,0.012987,0.012941,0.012891,0.012837,0.012779,0.012717,0.012652,0.012582,0.012509,0.012432,0.012351,0.012266,0.012178,0.012086,0.011990,0.011891,0.011788,0.011681,0.011571,0.011457,0.011340,0.011220,0.011095,0.010968,0.010837,0.010703,0.010566,0.010425,0.010282,0.010135,0.009985,0.009832,0.009676,0.009517,0.009355,0.009190,0.009023,0.008852,0.008680,0.008504,0.008326,0.008145,0.007962,0.007776,0.007588,0.007398,0.007205,0.007011,0.006814,0.006615,0.006414,0.006211,0.006006,0.005800,0.005591,0.005381,0.005169,0.004956,0.004741,0.004525,0.004307,0.004088,0.003868,0.003647,0.003424,0.003201,0.002976,0.002751,0.002524,0.002297,0.002070,0.001841,0.001612,0.001383,0.001153,0.000923,0.000692,0.000462,0.000231,0.000000,-0.000231,-0.000462,-0.000692,-0.000923,-0.001153,-0.001383,-0.001612,-0.001841,-0.002070,-0.002297,-0.002524,-0.002751,-0.002976,-0.003201,-0.003424,-0.003647,-0.003868,-0.004088,-0.004307,-0.004525,-0.004741,-0.004956,-0.005169,-0.005381,-0.005591,-0.005800,-0.006006,-0.006211,-0.006414,-0.006615,-0.006814,-0.007011,-0.007205,-0.007398,-0.007588,-0.007776,-0.007962,-0.008145,-0.008326,-0.008504,-0.008680,-0.008852,-0.009023,-0.009190,-0.009355,-0.009517,-0.009676,-0.009832,-0.009985,-0.010135,-0.010282,-0.010425,-0.010566,-0.010703,-0.010837,-0.010968,-0.011095,-0.011220,-0.011340,-0.011457,-0.011571,-0.011681,-0.011788,-0.011891,-0.011990,-0.012086,-0.012178,-0.012266,-0.012351,-0.012432,-0.012509,-0.012582,-0.012652,-0.012717,-0.012779,-0.012837,-0.012891,-0.012941,-0.012987,-0.013029,-0.013067,-0.013101,-0.013131,-0.013157,-0.013179,-0.013198,-0.013212,-0.013222,-0.013228,-0.013230,-0.013228,-0.013222,-0.013212,-0.013198,-0.013179,-0.013157,-0.013131,-0.013101,-0.013067,-0.013029,-0.012987,-0.012941,-0.012891,-0.012837,-0.012779,-0.012717,-0.012652,-0.012582,-0.012509,-0.012432,-0.012351,-0.012266,-0.012178,-0.012086,-0.011990,-0.011891,-0.011788,-0.011681,-0.011571,-0.011457,-0.011340,-0.011220,-0.011095,-0.010968,-0.010837,-0.010703,-0.010566,-0.010425,-0.010282,-0.010135,-0.009985,-0.009832,-0.009676,-0.009517,-0.009355,-0.009190,-0.009023,-0.008852,-0.008680,-0.008504,-0.008326,-0.008145,-0.007962,-0.007776,-0.007588,-0.007398,-0.007205,-0.007011,-0.006814,-0.006615,-0.006414,-0.006211,-0.006006,-0.005800,-0.005591,-0.005381,-0.005169,-0.004956,-0.004741,-0.004525,-0.004307,-0.004088,-0.003868,-0.003647,-0.003424,-0.003201,-0.002976,-0.002751,-0.002524,-0.002297,-0.002070,-0.001841,-0.001612,-0.001383,-0.001153,-0.000923,-0.000692,-0.000462,-0.000231,-0.000000,0.000231,0.000462,0.000692,0.000923,0.001153,0.001383,0.001612,0.001841,0.002070,0.002297,0.002524,0.002751,0.002976,0.003201,0.003424,0.003647,0.003868,0.004088,0.004307,0.004525,0.004741,0.004956,0.005169,0.005381,0.005591,0.005800,0.006006,0.006211,0.006414,0.006615,0.006814,0.007011,0.007205,0.007398,0.007588,0.007776,0.007962,0.008145,0.008326,0.008504,0.008680,0.008852,0.009023,0.009190,0.009355,0.009517,0.009676,0.009832,0.009985,0.010135,0.010282,0.010425,0.010566,0.010703,0.010837,0.010968,0.011095,0.011220,0.011340,0.011457,0.011571,0.011681,0.011788,0.011891,0.011990,0.012086,0.012178,0.012266,0.012351,0.012432,0.012509,0.012582,0.012652,0.012717,0.012779,0.012837,0.012891,0.012941,0.012987,0.013029,0.013067,0.013101,0.013131,0.013157,0.013179,0.013198,0.013212,0.013222,0.013228]

polar_pattern = [omni, homni, card, hypcard, subcard, bidir]
polar_pattern_names = ["Omni", "Half-Omni", "Cardioid",
                       "Hypercardioid", "Subcardioid", "Bidirectional"]


def create_polar_plot(i, fig, amps, title):
    print(f"{len(amps)}")
    theta = np.linspace(0, 2*np.pi, len(amps))  # angles
    ax = fig.add_subplot(2, 3, i+1, projection='polar')
    ax.plot(theta, amps)
    ax.grid(True)
    ax.set_title(title, va='center')


fig = plt.figure(1)
for i in range(0, 6):
    create_polar_plot(i, fig, polar_pattern[i], polar_pattern_names[i])

plt.show()
