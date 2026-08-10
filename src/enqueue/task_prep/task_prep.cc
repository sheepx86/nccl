/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "enqueue.h"
#include "config/algorithm_registry.h"

static ncclResult_t ncclTaskSelectSymA2a(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo) {
  struct ncclRawTaskColl* raw = &tInfo->raw->coll;
  uint64_t algMask = 0;
  NCCLCHECK(ncclCollConfigGetAlgMask(&raw->collConfig, raw->func, &algMask));
  uint64_t a2aMask = 1ull << (NCCL_TUNING_SYM_KERNEL_ID_OFFSET + ncclSymkKernelId_AlltoAll_FullGin_LsaST);
  bool selectionAllows = algMask == 0 || (algMask & a2aMask) != 0;
  bool ctaPolicyEnvOverridden = ncclGetEnvCtaPolicy() != NCCL_CONFIG_UNDEF_INT;
  int ctaPolicy =
    ncclCollConfigResolveCTAPolicy(raw->collConfig.CTAPolicy, comm->config.CTAPolicy, ctaPolicyEnvOverridden);
  size_t bytesPerPeer = raw->count * ncclTypeSize(raw->datatype);

  if (selectionAllows && !(ctaPolicy & NCCL_CTA_POLICY_ZERO) &&
      ncclSymkA2aAvailable(comm, raw->sendbuff, raw->recvbuff, bytesPerPeer)) {
    int nChannels = ncclSymkA2aChannels(comm, bytesPerPeer, tInfo->tuningIn.minCTAs, tInfo->tuningIn.maxCTAs);
    tInfo->tuningOut.id = NCCL_TUNING_SYM_KERNEL_ID_OFFSET + ncclSymkKernelId_AlltoAll_FullGin_LsaST;
    tInfo->tuningOut.valid = 1;
    tInfo->tuningOut.timeUs = 0;
    tInfo->tuningOut.symKernelId = ncclSymkKernelId_AlltoAll_FullGin_LsaST;
    tInfo->tuningOut.nChannels = nChannels;
    tInfo->tuningOut.maxChannels = nChannels;
    tInfo->tuningOut.nWarps = ncclSymkMaxThreads / WARP_SIZE;
  } else if (algMask != 0 && raw->collConfig.forceAlgSelection) {
    WARN("algSelection names only symmetric kernel(s) that are unavailable for AlltoAll");
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t ncclTaskPrepare(struct ncclComm* comm, ncclSimInfo_t* simInfo) {
  struct ncclTaskTuningInfo* tInfo;
  struct ncclTaskTuningInfoQueue taskTuningInfoQueue;

  ncclIntruQueueConstruct(&taskTuningInfoQueue.queue);
  NCCLCHECK(ncclTaskPreTuning(comm, &comm->rawTaskQueue, &taskTuningInfoQueue));

  tInfo = ncclIntruQueueHead(&taskTuningInfoQueue.queue);
  for (; tInfo != nullptr; tInfo = tInfo->next) {
    if (tInfo->raw->kind == ncclTaskKindColl && tInfo->tuningIn.func == ncclFuncAlltoAll) {
      NCCLCHECK(ncclTaskSelectSymA2a(comm, tInfo));
      continue;
    }
    // need tuning to support all functions.
    if (tInfo->raw->kind == ncclTaskKindColl && tInfo->tuningIn.func != ncclFuncScatter &&
        tInfo->tuningIn.func != ncclFuncGather &&
        tInfo->tuningIn.func != ncclFuncAllGatherV) {
      NCCLCHECK(ncclTuningCompute(&tInfo->tuningIn, &tInfo->tuningOut));
    }
  }

  NCCLCHECK(ncclTaskClassification(comm, &taskTuningInfoQueue, &comm->classifiedTaskQueues));

  NCCLCHECK(ncclTaskPostTuning(comm, &comm->classifiedTaskQueues, simInfo));

  return ncclSuccess;
}
