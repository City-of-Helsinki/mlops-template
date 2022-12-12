Store here temporary data & files that are used or created at runtime.
For example log and backup files. 
By default replaced by a Docker tmpfs storage - everything in this folder is lost when the container stops. This is to avoid collecting and forgetting personal data by accident. To save files persistently, edit storage type in compose.yml.