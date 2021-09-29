with (import <nixpkgs> {});


let
  pythonEnv = python38.withPackages (
    ps: [ ps.sympy
          ps.numpy
        ]);

in mkShell {
  buildInputs = [ pythonEnv
                ];
}
