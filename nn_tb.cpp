#include <iostream>
#include <fstream>
#include "nn.h"

static bool load_pixels(const char *path, pixel_t buf[INPUT_SIZE]) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cout << "  [SKIP] cannot open " << path << std::endl;
        return false;
    }
    for (int i = 0; i < INPUT_SIZE; i++) {
        int v;
        if (!(f >> v)) {
            std::cout << "  [ERROR] short read in " << path << std::endl;
            return false;
        }
        buf[i] = static_cast<pixel_t>(v & 0xFF);
    }
    return true;
}

static int check(const char *label, pixel_t buf[INPUT_SIZE], int expected) {
    ap_uint<1> result = 0;
    nn_top(buf, &result);
    int got  = (int)(unsigned)result;
    int pass = (got == expected);
    std::cout << label
              << " -> " << got
              << "  (expect " << expected << ")  "
              << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? 0 : 1;
}

struct TestCase { const char *file; int expected; const char *label; };

static const TestCase TESTS[] = {
    { "test_d_1.txt",     1, "d_1    " },
    { "test_d_2.txt",     1, "d_2    " },
    { "test_d_3.txt",     1, "d_3    " },
    { "test_d_4.txt",     1, "d_4    " },
    { "test_d_5.txt",     1, "d_5    " },
    { "test_d_6.txt",     1, "d_6    " },
    { "test_d_7.txt",     1, "d_7    " },
    { "test_d_8.txt",     1, "d_8    " },
    { "test_d_9.txt",     1, "d_9    " },
    { "test_d_10.txt",    1, "d_10   " },
    { "test_d_11.txt",    1, "d_11   " },
    { "test_d_12.txt",    1, "d_12   " },
    { "test_d_13.txt",    1, "d_13   " },
    { "test_d_14.txt",    1, "d_14   " },
    { "test_d_15.txt",    1, "d_15   " },
    { "test_d_16.txt",    1, "d_16   " },
    { "test_d_17.txt",    1, "d_17   " },
    { "test_d_18.txt",    1, "d_18   " },
    { "test_d_19.txt",    1, "d_19   " },
    { "test_d_20.txt",    1, "d_20   " },
    { "test_d_21.txt",    1, "d_21   " },
    { "test_d_22.txt",    1, "d_22   " },
    { "test_d_23.txt",    1, "d_23   " },
    { "test_d_24.txt",    1, "d_24   " },
    { "test_d_25.txt",    1, "d_25   " },
    { "test_d_26.txt",    1, "d_26   " },
    { "test_d_27.txt",    1, "d_27   " },
    { "test_d_28.txt",    1, "d_28   " },
    { "test_d_29.txt",    1, "d_29   " },
    { "test_d_30.txt",    1, "d_30   " },
    { "test_d_31.txt",    1, "d_31   " },
    { "test_d_32.txt",    1, "d_32   " },
    { "test_d_33.txt",    1, "d_33   " },
    { "test_d_34.txt",    1, "d_34   " },
    { "test_d_35.txt",    1, "d_35   " },
    { "test_d_36.txt",    1, "d_36   " },
    { "test_d_37.txt",    1, "d_37   " },
    { "test_d_38.txt",    1, "d_38   " },
    { "test_d_39.txt",    1, "d_39   " },
    { "test_d_40.txt",    1, "d_40   " },
    { "test_d_41.txt",    1, "d_41   " },
    { "test_d_42.txt",    1, "d_42   " },
    { "test_d_43.txt",    1, "d_43   " },
    { "test_d_44.txt",    1, "d_44   " },
    { "test_d_45.txt",    1, "d_45   " },
    { "test_d_46.txt",    1, "d_46   " },
    { "test_d_47.txt",    1, "d_47   " },
    { "test_d_48.txt",    1, "d_48   " },
    { "test_d_49.txt",    1, "d_49   " },
    { "test_d_50.txt",    1, "d_50   " },
    { "test_notd_1.txt",  0, "notd_1 " },
    { "test_notd_2.txt",  0, "notd_2 " },
    { "test_notd_3.txt",  0, "notd_3 " },
    { "test_notd_4.txt",  0, "notd_4 " },
    { "test_notd_5.txt",  0, "notd_5 " },
    { "test_notd_6.txt",  0, "notd_6 " },
    { "test_notd_7.txt",  0, "notd_7 " },
    { "test_notd_8.txt",  0, "notd_8 " },
    { "test_notd_9.txt",  0, "notd_9 " },
    { "test_notd_10.txt", 0, "notd_10" },
    { "test_notd_11.txt", 0, "notd_11" },
    { "test_notd_12.txt", 0, "notd_12" },
    { "test_notd_13.txt", 0, "notd_13" },
    { "test_notd_14.txt", 0, "notd_14" },
    { "test_notd_15.txt", 0, "notd_15" },
    { "test_notd_16.txt", 0, "notd_16" },
    { "test_notd_17.txt", 0, "notd_17" },
    { "test_notd_18.txt", 0, "notd_18" },
    { "test_notd_19.txt", 0, "notd_19" },
    { "test_notd_20.txt", 0, "notd_20" },
    { "test_notd_21.txt", 0, "notd_21" },
    { "test_notd_22.txt", 0, "notd_22" },
    { "test_notd_23.txt", 0, "notd_23" },
    { "test_notd_24.txt", 0, "notd_24" },
    { "test_notd_25.txt", 0, "notd_25" },
    { "test_notd_26.txt", 0, "notd_26" },
    { "test_notd_27.txt", 0, "notd_27" },
    { "test_notd_28.txt", 0, "notd_28" },
    { "test_notd_29.txt", 0, "notd_29" },
    { "test_notd_30.txt", 0, "notd_30" },
    { "test_notd_31.txt", 0, "notd_31" },
    { "test_notd_32.txt", 0, "notd_32" },
    { "test_notd_33.txt", 0, "notd_33" },
    { "test_notd_34.txt", 0, "notd_34" },
    { "test_notd_35.txt", 0, "notd_35" },
    { "test_notd_36.txt", 0, "notd_36" },
    { "test_notd_37.txt", 0, "notd_37" },
    { "test_notd_38.txt", 0, "notd_38" },
    { "test_notd_39.txt", 0, "notd_39" },
    { "test_notd_40.txt", 0, "notd_40" },
    { "test_notd_41.txt", 0, "notd_41" },
    { "test_notd_42.txt", 0, "notd_42" },
    { "test_notd_43.txt", 0, "notd_43" },
    { "test_notd_44.txt", 0, "notd_44" },
    { "test_notd_45.txt", 0, "notd_45" },
    { "test_notd_46.txt", 0, "notd_46" },
    { "test_notd_47.txt", 0, "notd_47" },
    { "test_notd_48.txt", 0, "notd_48" },
    { "test_notd_49.txt", 0, "notd_49" },
    { "test_notd_50.txt", 0, "notd_50" },
};

static const int N_TESTS = sizeof(TESTS) / sizeof(TESTS[0]);

int main() {
    static pixel_t buf[INPUT_SIZE];
    int failures = 0, skipped = 0;

    std::cout << "==============================" << std::endl;
    std::cout << "  NN Testbench (" << N_TESTS << " cases)" << std::endl;
    std::cout << "==============================" << std::endl;

    for (int t = 0; t < N_TESTS; t++) {
        if (!load_pixels(TESTS[t].file, buf)) { skipped++; continue; }
        failures += check(TESTS[t].label, buf, TESTS[t].expected);
    }

    std::cout << "------------------------------" << std::endl;
    std::cout << "Passed  : " << (N_TESTS - skipped - failures) << std::endl;
    std::cout << "Failed  : " << failures  << std::endl;
    std::cout << "Skipped : " << skipped   << std::endl;
    std::cout << "==============================" << std::endl;

    return failures;
}