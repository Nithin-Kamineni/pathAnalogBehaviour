// File: 23_bit_adder.v
module 23_bit_adder (
    input a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, cin;
    output sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16, sum17, sum18, sum19, sum20, sum21, sum22, cout;
);

wire c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22;

// Bit 0
assign sum0 = a0 ^ b0 ^ cin;
assign c1 = (a0 & b0) | (b0 & cin) | (a0 & cin);

// Bit 1
assign sum1 = a1 ^ b1 ^ c1;
assign c2 = (a1 & b1) | (b1 & c1) | (a1 & c1);

// Bit 2
assign sum2 = a2 ^ b2 ^ c2;
assign c3 = (a2 & b2) | (b2 & c2) | (a2 & c2);

// Bit 3
assign sum3 = a3 ^ b3 ^ c3;
assign c4 = (a3 & b3) | (b3 & c3) | (a3 & c3);

// Bit 4
assign sum4 = a4 ^ b4 ^ c4;
assign c5 = (a4 & b4) | (b4 & c4) | (a4 & c4);

// Bit 5
assign sum5 = a5 ^ b5 ^ c5;
assign c6 = (a5 & b5) | (b5 & c5) | (a5 & c5);

// Bit 6
assign sum6 = a6 ^ b6 ^ c6;
assign c7 = (a6 & b6) | (b6 & c6) | (a6 & c6);

// Bit 7
assign sum7 = a7 ^ b7 ^ c7;
assign c8 = (a7 & b7) | (b7 & c7) | (a7 & c7);

// Bit 8
assign sum8 = a8 ^ b8 ^ c8;
assign c9 = (a8 & b8) | (b8 & c8) | (a8 & c8);

// Bit 9
assign sum9 = a9 ^ b9 ^ c9;
assign c10 = (a9 & b9) | (b9 & c9) | (a9 & c9);

// Bit 10
assign sum10 = a10 ^ b10 ^ c10;
assign c11 = (a10 & b10) | (b10 & c10) | (a10 & c10);

// Bit 11
assign sum11 = a11 ^ b11 ^ c11;
assign c12 = (a11 & b11) | (b11 & c11) | (a11 & c11);

// Bit 12
assign sum12 = a12 ^ b12 ^ c12;
assign c13 = (a12 & b12) | (b12 & c12) | (a12 & c12);

// Bit 13
assign sum13 = a13 ^ b13 ^ c13;
assign c14 = (a13 & b13) | (b13 & c13) | (a13 & c13);

// Bit 14
assign sum14 = a14 ^ b14 ^ c14;
assign c15 = (a14 & b14) | (b14 & c14) | (a14 & c14);

// Bit 15
assign sum15 = a15 ^ b15 ^ c15;
assign c16 = (a15 & b15) | (b15 & c15) | (a15 & c15);

// Bit 16
assign sum16 = a16 ^ b16 ^ c16;
assign c17 = (a16 & b16) | (b16 & c16) | (a16 & c16);

// Bit 17
assign sum17 = a17 ^ b17 ^ c17;
assign c18 = (a17 & b17) | (b17 & c17) | (a17 & c17);

// Bit 18
assign sum18 = a18 ^ b18 ^ c18;
assign c19 = (a18 & b18) | (b18 & c18) | (a18 & c18);

// Bit 19
assign sum19 = a19 ^ b19 ^ c19;
assign c20 = (a19 & b19) | (b19 & c19) | (a19 & c19);

// Bit 20
assign sum20 = a20 ^ b20 ^ c20;
assign c21 = (a20 & b20) | (b20 & c20) | (a20 & c20);

// Bit 21
assign sum21 = a21 ^ b21 ^ c21;
assign c22 = (a21 & b21) | (b21 & c21) | (a21 & c21);

// Bit 22
assign sum22 = a22 ^ b22 ^ c22;
assign cout = (a22 & b22) | (b22 & c22) | (a22 & c22);

endmodule
