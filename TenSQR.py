#!/usr/bin/env python3

import argparse
import math
import random
import sys
import time

import numpy as np
import scipy as sp
from scipy.stats import binom


def ACGT_count(M_E):
    """
    ACGT statistics of a read matrix.

    Args:
        M_E: (n_reads, n_SNVs) array

    Returns:
        (n_SNVs, 4) array containing the count of each A, C, G and T in each SNV.
    """
    if M_E.ndim != 2:
        raise ValueError("arr should be 2D")
    n_snvs = M_E.shape[1]
    out = np.zeros((n_snvs, 4))
    for i in range(4):
        out[:, i] = (M_E == (i + 1)).sum(axis=0)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TenSQR.py")
    parser.add_argument(
        "--zone_name",
        help=(
            "Prefix for outputs. Must match prefix of 'SNV_pos.txt', 'SNV_matrix.txt', "
            "'lowQSseq.txt' and 'Homo_seq.txt'"
        ),
        required=True,
    )
    parser.add_argument(
        "--err_rate",
        type=float,
        help="Sequencing error rate (percent) (default=0.2).",
        default=0.2,
        required=False,
    )
    parser.add_argument(
        "--MEC_thre",
        type=float,
        help="MEC improvement threshold (default=0.0312).",
        default=0.0312,
        required=False,
    )
    parser.add_argument(
        "--K",
        type=float,
        help="Initial population size (default=5).",
        default=5,
        required=False,
    )
    args = parser.parse_args()

    zone_name = args.zone_name
    err_rate = args.err_rate
    MEC_thre = args.MEC_thre
    K = args.K

    # prameter setting
    seq_err = err_rate / 100  # sequencing error rate^M
    p_value = 10**-5  # p-value^M

    # name of imported data
    SNVposname = zone_name + "_SNV_pos.txt"  # SNV position name
    SNVmatrixname = zone_name + "_SNV_matrix.txt"  # SNV matrix name
    lowQSseqname = zone_name + "_lowQSseq.txt"  # low quality score sequence name
    Homoseqname = zone_name + "_Homo_seq.txt"  # homo sequence name

    # import SNV matrix
    SNVmatrix = np.loadtxt(SNVmatrixname, ndmin=2).astype(int)

    if len(SNVmatrix) == 0:
        print(
            f"SNV matrix ({SNVmatrixname}) is empty. Not running QSR.", file=sys.stderr
        )
        exit()

    ori_ACGTcount = ACGT_count(SNVmatrix)  # original ACGT statistics

    # import SNV position
    SNVpos = np.loadtxt(SNVposname).astype(int)

    if len(SNVpos) != SNVmatrix.shape[1]:
        raise ValueError(
            f"{SNVposname} and {SNVmatrixname} have different numbers of SNVs.\n"
            "Number of columns in the SNV matrix should match the number of SNV positions."
        )

    tStart = time.time()  # starting time

    # threshold for read assignment of the most dominant haplotype based on p-value
    num_read, hap_len = SNVmatrix.shape  # number of reads, length of haplotypes
    P_matrix = np.double(SNVmatrix != 0)  # projection matrix
    P_tensor = np.tile(P_matrix, (1, 4))  # projection matrix of tensor structure
    nongap = P_matrix.sum(axis=1)  # number of nongap positions of each read
    max_thre = 20  # maximum mismatches for a read and a haplotype # 7 300
    max_len = 300  # maximum number of nongap positions
    L = []  # threshold for number of nongap positions
    Th = []  # corresponding maximum number of mismatches
    for thre in range(1, max_thre + 1):
        for l in range(1, max_len + 1):
            pr = 1
            for k in range(thre):
                pr -= binom.pmf(k, l, seq_err)
            if pr >= p_value:
                Th.append(thre)
                L.append(l)
                break
    L[0] += 1
    mis_cri = np.zeros((num_read), dtype=int)  # criteria of mismatches for each read
    for i in range(num_read):
        for l in range(len(Th)):
            if nongap[i] < L[l]:
                mis_cri[i] = l + 1
                break
        if mis_cri[i] == 0:
            mis_cri[i] = len(Th) + 1
    ori_mis_cri = mis_cri.copy()  # original criteria of mismatches for each read
    ori_num_read = num_read  # original number of reads

    # rank estimation+successive clustering+alternating minimization parameter setting
    error_thre = 10**-5  # stopping criteria for alternating minimization
    max_ite = 2000  # maximum iteration number for alternation minimization
    K_table = np.array([1, 0])  # K table to decide whether to stop rank estimation
    K_step = int(K)  # searching step for rank estimation
    K_count = 1  # count the number of rank searching
    MEC_table = np.zeros(
        (5, 50)
    )  # table to record K(set length to 50), returned number of haplotypes, recall, MEC and MEC rate

    # rank estimation
    while K_table[1] - K_table[0] != 1:  # stopping criteria for rank estimation
        K = int(K)
        for K_ite in range(
            K, K + 2
        ):  # search 2 continuous K values to calculate MEC rate
            if len(np.where(MEC_table[0, :] == K_ite)[0]) == 0:
                alt_tag1 = 1
                # indicator for plus sign alternating minimization
                alt_tag2 = 1
                # indicator for minus sign alternating minimization
                MEC = np.array(
                    [np.inf, np.inf]
                )  # MEC to record alternating minimization with different signs
                recall = np.zeros(2)  # record recall rate
                hap_num = np.zeros(2)  # number of haplotypes reconstructed
                for svd_flag in range(1, 3):  # 1 for plus sign; 2 for minus sign
                    print("K_ite = " + str(K_ite))
                    print("svd_flag = " + str(svd_flag))
                    R = K_ite
                    M_E = SNVmatrix.copy()  # read matrix
                    TM_E = np.concatenate(
                        (
                            np.double(M_E == 1),
                            np.double(M_E == 2),
                            np.double(M_E == 3),
                            np.double(M_E == 4),
                        ),
                        axis=1,
                    )  # read matrix in tensor structure
                    mis_cri = ori_mis_cri.copy()  # criteria of mismatches for each read
                    ori_K = R  # original K value
                    num_V = 1
                    reconV = np.zeros(
                        (R, hap_len), dtype=int
                    )  # reconstructed haplotypes
                    # successive clustering
                    while R != 0 and len(M_E[:, 0]) > R:
                        print("R = " + str(R))
                        P_matrix = np.double(M_E != 0)  # updated projection matrix
                        P_tensor = np.tile(
                            P_matrix, (1, 4)
                        )  # updated projection matrix of tensor structure
                        num_read = len(M_E[:, 0])  # updated number of read
                        Ut, S, Vt = sp.sparse.linalg.svds(TM_E, R)  # svd
                        Vt = np.dot(
                            np.diag(np.sqrt(S)), Vt
                        )  # initial real-valued haplotypes
                        Vt = Vt[::-1]
                        if svd_flag == 2:
                            Vt = -Vt
                        ACGTcount = ACGT_count(M_E)  # updated ACGT statistics
                        BV = np.eye(R, dtype=int)  # Basic vectors of dimension R
                        ite = 0  # iteration count
                        err = np.inf  # current Frobenius norm
                        err_Com = (
                            np.inf
                        )  # difference between current and previous Frobenius norm
                        err_hap = (
                            np.inf
                        )  # Frobenius norm of the difference between current and previous haplotypes
                        err_hist = np.zeros(
                            (1, max_ite)
                        )  # record current Frobenius norm
                        Vt_last = 100 * np.ones(
                            (R, 4 * hap_len)
                        )  # initialization for haplotypes of last iteration

                        # alternating minimization
                        while (
                            err_hap > error_thre
                            and err > error_thre
                            and err_Com > error_thre
                            and ite < max_ite
                        ):
                            ite += 1
                            # update U matrix
                            U = np.zeros((num_read, R))
                            for i in range(R):
                                U[:, i] = np.sum(
                                    ((TM_E - Vt[i, :]) * P_tensor) ** 2, axis=1
                                )
                            min_index = np.argmin(U, axis=1)
                            U = BV[min_index.astype(int), :]
                            # update V matrix
                            V_major = np.zeros((R, hap_len))  # majority voting result
                            for i in range(R):
                                reads_single = M_E[
                                    min_index == i, :
                                ]  # all reads from one haplotypes
                                single_sta = np.zeros((hap_len, 4))
                                if len(reads_single) != 0:
                                    single_sta = ACGT_count(
                                        reads_single
                                    )  # ACGT statistics of a single nucleotide position
                                V_major[i, :] = np.argmax(single_sta, axis=1) + 1
                                uncov_pos = np.where(np.sum(single_sta, axis=1) == 0)[0]
                                for j in range(len(uncov_pos)):
                                    if (
                                        len(
                                            np.where(
                                                ACGTcount[uncov_pos[j], :]
                                                == max(ACGTcount[uncov_pos[j], :])
                                            )[0]
                                        )
                                        != 1
                                    ):  # if not covered, select the most doninant one based on 'ACGTcount'
                                        tem = np.where(
                                            ACGTcount[uncov_pos[j], :]
                                            == max(ACGTcount[uncov_pos[j], :])
                                        )[0]
                                        V_major[i, uncov_pos[j]] = (
                                            tem[
                                                int(
                                                    np.floor(random.random() * len(tem))
                                                )
                                            ]
                                            + 1
                                        )
                                    else:
                                        V_major[i, uncov_pos[j]] = (
                                            np.argmax(ACGTcount[uncov_pos[j], :]) + 1
                                        )
                            Vt = np.concatenate(
                                (
                                    np.double(V_major == 1),
                                    np.double(V_major == 2),
                                    np.double(V_major == 3),
                                    np.double(V_major == 4),
                                ),
                                axis=1,
                            )

                            # termination criteria
                            err = np.linalg.norm(
                                (TM_E - np.dot(U, Vt)) * P_tensor, ord="fro"
                            )
                            err_hist[0, ite - 1] = err
                            if ite > 1:
                                err_Com = abs(
                                    err_hist[0, ite - 1] - err_hist[0, ite - 2]
                                )
                            err_hap = np.linalg.norm(Vt - Vt_last, ord="fro") / np.sqrt(
                                4 * hap_len / R
                            )
                            Vt_last = Vt.copy()
                            print(
                                "ite: "
                                + str(ite)
                                + "; err: "
                                + str(err)
                                + "; err_Com: "
                                + str(err_Com)
                                + "; err_hap: "
                                + str(err_hap)
                                + "; R: "
                                + str(R)
                            )

                        V = np.argmax(Vt.reshape(R, hap_len, 4, order="F"), axis=2) + 1

                        # assign reads to the most dominant haplotype
                        domi_flag = np.argmax(
                            U.sum(axis=0)
                        )  # the index of the most dominant haplotype
                        V_flag = V[domi_flag, :]
                        HD_table = np.zeros(
                            (num_read, 3)
                        )  # a table to record 'number of identical nucleotides', 'number of nongap positions' and 'hamming distance'
                        index = (
                            []
                        )  # indices for reads to be assigned to the most dominant haplotype
                        num_mem = 0  # count the number of assigned reads
                        HD_table[:, 0] = ((M_E - V[domi_flag, :]) == 0).sum(
                            axis=1
                        )  # number of identical nucleotides
                        HD_table[:, 1] = (M_E != 0).sum(
                            axis=1
                        )  # number of nongap positions
                        HD_table[:, 2] = (
                            HD_table[:, 1] - HD_table[:, 0]
                        )  # hamming distance

                        for i in range(num_read):
                            if (
                                HD_table[i, 2] == 0
                            ):  # assign the read if hamming distance is 0
                                num_mem += 1
                                index.append(i)
                            elif (
                                HD_table[i, 2] <= mis_cri[i]
                            ):  # if hamming distance is not 0, assign the read based on probability distributions
                                pos = np.where(M_E[i, :] != 0)[
                                    0
                                ]  # the position of nongaps
                                pr_variant = (
                                    1  # initial variant probability of the read
                                )
                                for j in range(len(pos)):
                                    pr_variant *= ACGTcount[
                                        pos[j], M_E[i, pos[j]] - 1
                                    ] / sum(ACGTcount[pos[j], :])
                                pr_seq = binom.pmf(
                                    HD_table[i, 2], HD_table[i, 1], seq_err
                                )  # sequencing error of the read
                                if pr_seq > pr_variant:
                                    num_mem += 1
                                    index.append(i)

                        # decide whether to stop current successive clustering
                        if len(index) == 0:
                            if svd_flag == 1:
                                alt_tag1 = 0
                                break
                            else:
                                alt_tag2 = 0
                                break

                        # aditional majority voting
                        index = np.array(index)
                        addi_count = ACGT_count(
                            M_E[index, :]
                        )  # ACGT statistics for additional majority voting
                        V[domi_flag, :] = (
                            np.argmax(addi_count, axis=1) + 1
                        ) * np.double(
                            np.sum(addi_count, axis=1) != 0
                        ) + V_flag * np.double(
                            np.sum(addi_count, axis=1) == 0
                        )

                        # remove assigned reads
                        reconV[num_V - 1, :] = V[
                            domi_flag, :
                        ]  # record the most dominant haplotype
                        mis_cri = np.delete(
                            mis_cri, index, 0
                        )  # remove corresponding 'mis_cri'
                        M_E = np.delete(M_E, index, 0)  # remove reads
                        num_read = len(M_E[:, 0])  # update the number of reads
                        TM_E = np.concatenate(
                            (
                                np.double(M_E == 1),
                                np.double(M_E == 2),
                                np.double(M_E == 3),
                                np.double(M_E == 4),
                            ),
                            axis=1,
                        )  # update the read matrix in tensor structure
                        P_matrix = np.double(M_E != 0)  # updated projection matrix
                        P_tensor = np.tile(
                            P_matrix, (1, 4)
                        )  # updated projection matrix of tensor structure
                        num_V += 1
                        R -= 1

                    if (alt_tag1 == 1 and svd_flag == 1) or (
                        alt_tag2 == 1 and svd_flag == 2
                    ):
                        # one more majority voting after getting all the haplptypes
                        index = np.zeros(ori_num_read)  # indices for all the reads
                        # table of number of identical nucleotides
                        iden_table = np.zeros((ori_num_read, num_V - 1))
                        for i in range(num_V - 1):
                            iden_table[:, i] = (SNVmatrix - reconV[i, :] == 0).sum(
                                axis=1
                            )  # number of identical nucleotides for each read compared with the (i+1)th haplotype
                        index = np.argmax(iden_table, axis=1)
                        reconV2 = np.zeros(
                            (num_V - 1, hap_len)
                        )  # new haplotypes after one more majority voting
                        for i in range(num_V - 1):
                            reads_single = SNVmatrix[
                                index == i, :
                            ]  # all reads from one haplotypes
                            single_sta = np.zeros((hap_len, 4))
                            if len(reads_single) != 0:
                                single_sta = ACGT_count(
                                    reads_single
                                )  # ACGT statistics of a single nucleotide position
                            reconV2[i, :] = np.argmax(single_sta, axis=1) + 1
                            uncov_pos = np.where(np.sum(single_sta, axis=1) == 0)[0]
                            for j in range(len(uncov_pos)):
                                if (
                                    len(
                                        np.where(
                                            ori_ACGTcount[uncov_pos[j], :]
                                            == max(ori_ACGTcount[uncov_pos[j], :])
                                        )[0]
                                    )
                                    != 1
                                ):  # if not covered, select the most doninant one based on 'ACGTcount'
                                    tem = np.where(
                                        ori_ACGTcount[uncov_pos[j], :]
                                        == max(ori_ACGTcount[uncov_pos[j], :])
                                    )[0]
                                    reconV2[i, uncov_pos[j]] = (
                                        tem[int(np.floor(random.random() * len(tem)))]
                                        + 1
                                    )
                                else:
                                    reconV2[i, uncov_pos[j]] = (
                                        np.argmax(ori_ACGTcount[uncov_pos[j], :]) + 1
                                    )

                                    # MEC for reconV2
                        num_read = ori_num_read
                        true_ind = np.zeros(num_read)  # final indices of reads
                        iden_table = np.zeros(
                            (num_read, len(reconV2[:, 0]))
                        )  # table of number of identical nucleotides
                        for i in range(len(reconV2[:, 0])):
                            iden_table[:, i] = (SNVmatrix - reconV2[i, :] == 0).sum(
                                axis=1
                            )  # number of identical nucleotides for each read compared with the (i+1)th haplotype
                        true_ind = np.argmax(iden_table, axis=1)
                        M = reconV2[true_ind, :]  # Completed read matrix
                        P_matrix = SNVmatrix.copy()
                        P_matrix[P_matrix != 0] = 1  # projection matrix
                        MEC[svd_flag - 1] = len(
                            np.where((SNVmatrix - M) * P_matrix != 0)[0]
                        )
                        hap_num[svd_flag - 1] = len(
                            reconV2[:, 0]
                        )  # number of haplotypes returned

                        # record reconV2
                        if svd_flag == 1:
                            reconV3 = reconV2.copy()
                        else:
                            reconV4 = reconV2.copy()

                # break if alternating minimization does not work^M
                if alt_tag1 == 0 and alt_tag2 == 0:
                    K_count += 1
                    break

                # fill MEC_table
                MEC_table[0, K_count - 1] = ori_K  # original K
                MEC_index = np.argmin(MEC)
                MEC_table[1, K_count - 1] = hap_num[
                    MEC_index
                ]  # number of haplotypes returned
                MEC_table[2, K_count - 1] = min(MEC)  # smaller MEC
                MEC_table[3, K_count - 1] = recall[
                    MEC_index
                ]  # corresponding recall rate
                K_count += 1

                # record reconV2
                if MEC_index == 0:
                    reconV5 = reconV3.copy()
                else:
                    reconV5 = reconV4.copy()
                exec("reconVK" + str(K_ite) + " = reconV5")
            else:
                MEC_table[:, K_count - 1] = MEC_table[
                    :, np.where(MEC_table[0, :] == K_ite)[0][0]
                ]
                K_count += 1

        # rank estimation details
        if alt_tag1 == 0 and alt_tag2 == 0:
            MEC_table[0, K_count - 2] = ori_K
            K_table[0] = ori_K
            if K_table[1] == 0:
                K *= 2
            else:
                K = np.floor(sum(K_table) / 2)
        else:
            MEC_table[4, K_count - 3] = (
                MEC_table[2, K_count - 3] - MEC_table[2, K_count - 2]
            ) / MEC_table[
                2, K_count - 3
            ]  # MEC rate
            if MEC_table[4, K_count - 3] > MEC_thre:
                K_table[0] = MEC_table[0, K_count - 3]
                if math.log2(K_table[0] / K_step) % 1 == 0:
                    K *= 2
                else:
                    K = np.floor(sum(K_table) / 2)
            else:
                K_table[1] = MEC_table[0, K_count - 3]
                K = np.floor(sum(K_table) / 2)

    tEnd = time.time()
    i = np.where(MEC_table[0, :] == K_table[1])[0][0]
    # print('K = ' + str(MEC_table[0, i]))
    # print('MEC = ' + str(MEC_table[2, i]))
    # print('recall rate = ' + str(MEC_table[3, i]))
    print("MEC change rate = " + str(MEC_table[4, i]))
    print("CPU time: " + str(tEnd - tStart))

    # deletion
    (m, n) = eval("reconVK" + str(int(MEC_table[0, i])) + ".shape")
    reconV2 = eval("reconVK" + str(int(MEC_table[0, i])))
    index = np.zeros(ori_num_read)  # indices for all the reads
    iden_table = np.zeros((ori_num_read, m))  # table of number of identical nucleotides
    for i in range(m):
        iden_table[:, i] = (SNVmatrix - reconV2[i, :] == 0).sum(
            axis=1
        )  # number of identical nucleotides for each read compared with the (i+1)th haplotype
    index = np.argmax(iden_table, axis=1)

    V_deletion = np.zeros((m, n))
    for i in range(m):
        reads_single = SNVmatrix[index == i, :]  # all reads from one haplotype
        single_sta = np.zeros((hap_len, 4))
        if len(reads_single) != 0:
            single_sta = ACGT_count(
                reads_single
            )  # ACGT statistics of a single nucleotide position
        V_deletion[i, :] = np.argmax(single_sta, axis=1) + 1
        uncov_pos = np.where(np.sum(single_sta, axis=1) == 0)[0]
        if len(uncov_pos) != 0:
            V_deletion[i, uncov_pos] = 0
    V_deletion = V_deletion.astype(int)
    fre_count = []

    for i in range(m):
        fre_count.append((index == i).sum())

    # reorder haplotypes according to frequency
    viralseq_fre = fre_count / sum(fre_count)
    viralfre_index = np.argsort(viralseq_fre)
    viralfre_index = viralfre_index[::-1]
    m = np.linalg.matrix_rank(V_deletion)
    print(" ")
    print("Estimated population size : " + str(m))
    V_deletion_new = V_deletion[viralfre_index[:m]]
    viralseq_fre_new = []
    for i in range(m):
        tem = 0
        for j in range(len(V_deletion)):
            if (V_deletion_new[i, :] - V_deletion[j, :] != 0).sum() == 0:
                tem += viralseq_fre[j]
        viralseq_fre_new.append(tem)

    # homosequence
    Homoseq = np.loadtxt(Homoseqname)
    Homoseq = Homoseq.astype(int)
    Glen = len(Homoseq)
    K = m

    # full sequence
    Recon_Quasi = []
    for _ in range(K):
        Recon_Quasi.append(list(Homoseq))
    Recon_Quasi = np.array(Recon_Quasi)
    for i in range(Recon_Quasi.shape[0]):
        Recon_Quasi[i, SNVpos] = V_deletion_new[i, :]

    # output
    Quasi = []
    for i in range(len(Recon_Quasi)):
        seq = ""
        for j in range(len(Recon_Quasi[i, :])):
            if j + 1 > Glen and Recon_Quasi[i, j] == 0:
                break
            if Recon_Quasi[i, j] == 1:
                seq += "A"
            elif Recon_Quasi[i, j] == 2:
                seq += "C"
            elif Recon_Quasi[i, j] == 3:
                seq += "G"
            elif Recon_Quasi[i, j] == 4:
                seq += "T"
            elif Recon_Quasi[i, j] == 0:
                seq += "*"
        Quasi.append(seq)

    viralseq_fre = viralseq_fre_new
    Hashtable = dict()
    for i in range(len(Quasi)):
        if Quasi[i] not in Hashtable:
            Hashtable[Quasi[i]] = viralseq_fre[i]
        else:
            Hashtable[Quasi[i]] += viralseq_fre[i]

    Key = list(Hashtable.keys())
    Value = list(Hashtable.values())
    index = np.argsort(np.array(Value))[::-1]

    with open(zone_name + "_ViralSeq.fasta", "w") as f:
        for i in range(len(index)):
            freq = Value[index[i]]
            sequence = Key[index[i]]
            print(f"Frequency of strain {i + 1}: {freq:f}")
            f.write(f">strain_{i + 1} freq: {freq:f}\n")
            f.write(f"{sequence}\n")
