{
  description = "C++ dev";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.utils = {
    url = "github:numtide/flake-utils";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = inputs@{ self, nixpkgs, ... }:
    inputs.utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              linuxPackages = prev.linuxPackages.extend (selfLinux: superLinux: {
                nvidia_x11 = superLinux.nvidia_x11.overrideAttrs (s: rec {
                  version = "510.39.01";
                  name = (builtins.parseDrvName s.name).name + "-" + version;
                  src = prev.fetchurl {
                    url = "https://download.nvidia.com/XFree86/Linux-x86_64/${version}/NVIDIA-Linux-x86_64-${version}.run";
                    sha256 = "06wr3dfmmm4km8mcz56rzlz1r6fbk0n2570wp5g0m155zcxdqgif";
                  };
                });
              });
            })
          ];
          config.allowUnfree = true;
        };
      in
      {
        devShell = pkgs.mkShell rec {
          packages = with pkgs; [
            gcc
            cmake
            cmakeCurses
            opencv
            cudaPackages.cudatoolkit
            linuxPackages.nvidia_x11
          ];
          shellHook = ''
            export CUDA_PATH=${pkgs.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
            export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib:$EXTRA_LDFLAGS"
            export EXTRA_CCFLAGS="-I/usr/include"
          '';
        };
      });
}
