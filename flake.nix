{
  description = "vulkan rust dev environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          libxkbcommon

          vulkan-headers
          vulkan-loader
          vulkan-tools
          vulkan-validation-layers
          wayland

          shaderc

          cloc
        ];
        shellHook = ''
          export VK_LAYER_PATH=${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d
          export SHELL=${pkgs.zsh}/bin/zsh
          export LD_LIBRARY_PATH=${pkgs.wayland}/lib:$LD_LIBRARY_PATH
          export LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH
          export LD_LIBRARY_PATH=${pkgs.libxkbcommon}/lib:$LD_LIBRARY_PATH
          export SHADERC_LIB_DIR=${pkgs.shaderc.lib}/lib
          zsh
        '';
      };
    };
}
