// SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <iomanip>
#include "matrix.hpp"

int main()
{
    auto mat1 = Matrix<int>(5,5);
    auto mat2 = Matrix<int>(5,5);

    for (int i=0; i < 5; ++i)
        for (int j=0; j < 5; ++j) {
            mat1(i,j) = (i + 1) * 10 + j + 1;        
            mat2(i,j) = 0.0;
        }

    // std::copy(mat1.begin(0,0), mat1.end(1, 2), mat2.begin(0,0));
    std::copy(std::begin(mat1(0,0)), std::end(mat1(1, 2)), std::begin(mat2(0,0)));

    std::cout << "Matrix2" << std::endl;
    for (int i=0; i < 5; ++i) {
        for (int j=0; j < 5; ++j) {
            std::cout << std::setw(2)<< std::setfill('0') << mat2(i, j) << " ";
        }
        std::cout << std::endl;
    }
}


