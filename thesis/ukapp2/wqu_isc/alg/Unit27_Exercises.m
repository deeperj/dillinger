% The standard bit matrices
ZERO = [1;0];
ONE = [0;1];

% Hadamard gate (page 5 of lecture notes)
Hadamard = (1/sqrt(2))*[1,1;1,-1];

% Square root of NOT (page 6)
sqrtNOT = (1/sqrt(2))*[1,-1;1,1];

% The Pauli gates (page 6)
PauliX = [0,1;1,0];
PauliY = [0,-1i;1i,0];
PauliZ = [1,0;0,-1];

%{
A Matlab definition can be parameterise using the @(...) notation.
The definition below defines the general rotational gate.
@param vector A unit vector in the Bloch sphere.  A unit vector is
              a 3*1 matrix [x;y;z] such that the sum of the squares
              of the magnitudes of x, y, and z is one.  It defines
              a line reaching from the origin to point (x,y,z) on
              the Bloch sphere.  Here it defines the line around
              which we wish to rotate the Bloch sphere.
@param theta  The angle by which we want the Bloch sphere to be rotated.
              The angle is given in radians.  There are 2*pi radians
              in a full circle.
See page 6 of the lecture notes.
eye(2) is the 2 by 2 identity matrix.
vector(1), vector(2), and vector(3) access the first, second, and
third components of vector.
The "..." notation allows the definition to be continued onto a
new line.
%}
rotate = @(vector,theta) ...
        cos(theta/2)*eye(2) ...
        - 1i*sin(theta/2)*(vector(1)*PauliX ...
        + vector(2)*PauliY ...
        + vector(3)*PauliZ);

%{
The following unit vectors define the x, y, and z axes.
%}
Xaxis = [1;0;0];
Yaxis = [0;1;0];
Zaxis = [0;0;1];

% Rotate around the axes (page 6).
rotateX = @(theta) rotate(Xaxis,theta);
rotateY = @(theta) rotate(Yaxis,theta);
rotateZ = @(theta) rotate(Zaxis,theta);

% Phase shift (page 6).
phaseShift = @(theta) exp(1i*theta/2)*rotateZ(theta);

%{
A generic controlled U gate.  U should be a matrix for an n-qubit
gate, so it must be a 2^n*2^n matrix.
A generic controlled U gate for an n-qubit gate is a 2*2^n by 2*2^n
matrix (i.e. it is twice the height and width  of the original 
n-qubit gate).  The top right and bottom left quadrants are all 
zeros. The top left quadrant is a 2^n*2^n identity matrix, and the 
bottom right matrix is the original n-qubit matrix.
This is constructed using the following Matlab functions:
  - kron([0,0;0,1],U) will produce a matrix twice the height and
    width of U, with all entries except the bottom right quadrant
    zeros, and with the bottom right quadrant being a copy of U.
  - size(U,1) returns the number of rows in the matrix U
  - eye(n) returns the identity matrix of size n
  - so (eye(size(U,1)) returns an identity matrix of the same size
    as U
  - and kron([1,0;0,0],eye(size(U,1))) produces a matrix twice the
    height and width of U, with all entries except the top left
    quadrant zeros, and the top left quadrant being an identity
    matrix of the same size as U.
  - Adding these two matrices together gives the controlled U gate.
%}
c = @(U) kron([1,0;0,0],eye(size(U,1)))+kron([0,0;0,1],U);

%{
We can use this to define a controlled NOT gate.
First, define a NOT gate.
%}
NOT = [0,1;1,0];

% Now define a controlled NOT (page 6)
cNOT = c(NOT);

% A Toffoli gate is a controlled controlled NOT
Toffoli = c(cNOT);

% Deutsch's circuit...

% The U(f(n,m)) gate.
Uf = @(n,m) [~n,n,0,0;n,~n,0,0;0,0,~m,m;0,0,m,~m];

%Deutsch's circuit.
Deutsch = @(n,m) kron(Hadamard,Identity)*Uf(n,m)*kron(Hadamard,Hadamard);

