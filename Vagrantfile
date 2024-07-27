DOCKER_BUILD_DIR = ENV['DOCKER_BUILD_DIR']
DOCKER_FILE_PATH = ENV['DOCKER_FILE_PATH']
DOCKER_IMAGE_URI = ENV['DOCKER_IMAGE_URI']

OUTPUT_DIR_HOST = ENV['OUTPUT_DIR_HOST']
OUTPUT_DIR_GUEST = ENV['OUTPUT_DIR_GUEST']
OUTPUT_FILENAME = ENV['OUTPUT_FILENAME']

OUTPUT_TMP_HOST = ENV['OUTPUT_TMP_HOST']

Vagrant.configure("2") do |config|
  config.vm.box = "hashicorp/bionic64"
  config.vm.synced_folder "#{OUTPUT_DIR_HOST}", "#{OUTPUT_DIR_GUEST}", id: "core"
  config.vm.synced_folder "#{OUTPUT_TMP_HOST}", "/var/lib/docker/tmp", id: "temp"
  config.vm.provider "virtualbox" do |v|
    v.memory = 32384
    v.cpus = 8
  end

  config.vm.provision "docker" do |d|
    d.build_image "--rm -t #{DOCKER_IMAGE_URI} -f #{DOCKER_FILE_PATH} #{DOCKER_BUILD_DIR}"
  end

  config.vm.provision "shell",
    inline: "docker save #{DOCKER_IMAGE_URI} | gzip > #{OUTPUT_DIR_GUEST}/#{OUTPUT_FILENAME}"

end
