# Najlepiej przed pullem zmiany w run.sh i upewnić się że nie masz żadnych jobów w kolejce
# cd ~/Universal_AVR_Model
cp -r model_checkpoints/* /mnt/evafs/groups/mandziuk-lab/akaminski/model_checkpoints

# Opcjonalnie archiwizacja folderu z modelami przez usunięciem
# tar -czvf model_checkpoints_20240902.tar.gz model_checkpoints/
# mv model_checkpoints_20240902.tar.gz /mnt/evafs/groups/mandziuk-lab/akaminski
# rm -r model_checkpoints

# /mnt/evafs/groups/mandziuk-lab/akaminski/model_checkpoints/839928
# To coś ważnego? Bo mam u siebie w folderze a widzę też w grupie? (jeśli nie to dasz radę usunąć/zmienić nazwę bo mam tam model z którego korzystam (czy to po prostu kopia (niekompletna bo tylko last.ckpt jest w środku)))

cp -r logs/* /mnt/evafs/groups/mandziuk-lab/akaminski/logs
# tar -czvf logs_20240902.tar.gz logs/
# mv logs_20240902.tar.gz /mnt/evafs/groups/mandziuk-lab/akaminski
# rm -r logs

# Useful to cleanup ~/VirtualBox\ VMs/ folder (left overs from building docker/singularity image)
