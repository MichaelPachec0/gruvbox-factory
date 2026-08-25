{
  description = "gruvbox-factory: convert images, GIFs and video to the gruvbox palette";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      # The interpreters the CI matrix covers. These attribute names are the
      # package names CI builds: .#py313, .#py314, .#py314t
      interpreters = {
        py313 = pkgs: pkgs.python313;
        py314 = pkgs: pkgs.python314;
        py314t = pkgs: pkgs.python314FreeThreading;
      };

      mkPythonSet =
        system: python:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
            ]
          );

      mkEnvs =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        lib.mapAttrs (
          name: getPython:
          (mkPythonSet system (getPython pkgs)).mkVirtualEnv
            "gruvbox-factory-${name}-env"
            workspace.deps.default
        ) interpreters;
    in
    {
      packages = forAllSystems (
        system:
        let
          envs = mkEnvs system;
        in
        envs // { default = envs.py313; }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/gruvbox-factory";
        };
      });

      checks = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        (mkEnvs system)
        // {
          lint =
            pkgs.runCommandLocal "gruvbox-factory-ruff"
              {
                nativeBuildInputs = [ pkgs.ruff ];
              }
              ''
                cd ${self}
                ruff check --no-cache factory/
                touch $out
              '';
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = (mkPythonSet system pkgs.python313).overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "gruvbox-factory-dev-env" workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
              pkgs.ruff
              pkgs.ffmpeg
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
        }
      );

      overlays.default = final: _prev: {
        gruvbox-factory = self.packages.${final.system}.default;
      };
    };
}
