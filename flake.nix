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

      # The project plus its dev dependency group. workspace.deps.all enables
      # every optional-dependency and every dependency-group; deps.default
      # (used by packages.*) enables neither, so the shipped environment never
      # carries pytest.
      mkTestEnv =
        system: python:
        (mkPythonSet system python).mkVirtualEnv
          "gruvbox-factory-test-env"
          workspace.deps.all;

      # Run pytest over a writable copy of the flake source. The copy is
      # required: the source is a read-only store path, and pytest, numpy and
      # Pillow all want to write beside their inputs.
      mkPytestCheck =
        system: name: python: marker:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        pkgs.runCommandLocal "gruvbox-factory-${name}"
          {
            nativeBuildInputs = [ (mkTestEnv system python) ];
          }
          ''
            cp -r ${self} source
            chmod -R u+w source
            cd source
            export HOME="$TMPDIR/home"
            export XDG_CACHE_HOME="$TMPDIR/cache"
            export PYTHONDONTWRITEBYTECODE=1
            mkdir -p "$HOME" "$XDG_CACHE_HOME"
            pytest -m ${lib.escapeShellArg marker} tests/
            touch $out
          '';
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
        # The fast suite runs on every interpreter the build matrix covers,
        # including the free-threaded one. The goldens are numerically
        # identical on all three, so they run once.
        // (lib.mapAttrs' (
             name: getPython:
             lib.nameValuePair "tests-${name}" (
               mkPytestCheck system "tests-${name}" (getPython pkgs) "not slow"
             )
           ) interpreters)
        // {
          lint =
            pkgs.runCommandLocal "gruvbox-factory-ruff"
              {
                nativeBuildInputs = [ pkgs.ruff ];
              }
              ''
                cd ${self}
                ruff check --no-cache factory/ tests/
                touch $out
              '';

          golden = mkPytestCheck system "golden" pkgs.python313 "slow";
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
              # The slow tests build real 16 MiB tables. Run them on purpose:
              #   \pytest -m slow      (the backslash bypasses this alias)
              alias pytest="pytest -m 'not slow'"
            '';
          };
        }
      );

      overlays.default = final: _prev: {
        gruvbox-factory = self.packages.${final.system}.default;
      };
    };
}
