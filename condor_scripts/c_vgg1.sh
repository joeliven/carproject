universe=vanilla
Initialdir=/u/joeliven/repos/carproject
Executable=/u/joeliven/repos/carproject/condor_scripts/r_vgg1 .sh
+Group="GRAD"
+Project="INSTRUCTIONAL"
+ProjectDescription="Research"
+GPUJob=true
requirements=(TARGET.GPUSlot)
#requirements=(TARGET.GPUSlot && CUDAGlobalMemoryMb >= 6144)
#requirements=(TARGET.GPUSlot && TitanBlack == True)
#requirements=InMastodon #for non-GPU jobs
request_GPUs = 1
get_env=True
Notification=complete
Notify_user=joeliven@gmail.com
Log=/scratch/cluster/joeliven/carproject/logs/vgg1.log.$(Cluster)
Output=/scratch/cluster/joeliven/carproject/logs/vgg1.out.$(Cluster)
Error=/scratch/cluster/joeliven/carproject/logs/vgg1.err.$(Cluster)
Queue 1
