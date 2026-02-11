###############################################################
# CLOCK (100 MHz onboard oscillator)
###############################################################
set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -name sys_clk -period 10.000 [get_ports clk]


###############################################################
# RESET BUTTON (Center button → rst)
###############################################################
set_property PACKAGE_PIN U18 [get_ports rst]
set_property IOSTANDARD LVCMOS33 [get_ports rst]


###############################################################
# UART RX (USB-UART → FPGA)
###############################################################
set_property PACKAGE_PIN B18 [get_ports rx]
set_property IOSTANDARD LVCMOS33 [get_ports rx]


###############################################################
# 7-SEGMENT DISPLAY (Segments a–g)
# seg[0] = a
# seg[1] = b
# seg[2] = c
# seg[3] = d
# seg[4] = e
# seg[5] = f
# seg[6] = g
###############################################################
set_property PACKAGE_PIN W7 [get_ports {seg[0]}]
set_property PACKAGE_PIN W6 [get_ports {seg[1]}]
set_property PACKAGE_PIN U8 [get_ports {seg[2]}]
set_property PACKAGE_PIN V8 [get_ports {seg[3]}]
set_property PACKAGE_PIN U5 [get_ports {seg[4]}]
set_property PACKAGE_PIN V5 [get_ports {seg[5]}]
set_property PACKAGE_PIN U7 [get_ports {seg[6]}]

set_property IOSTANDARD LVCMOS33 [get_ports {seg[*]}]


###############################################################
# 7-SEGMENT DISPLAY ANODES
###############################################################
set_property PACKAGE_PIN U2 [get_ports {an[0]}]
set_property PACKAGE_PIN U4 [get_ports {an[1]}]
set_property PACKAGE_PIN V4 [get_ports {an[2]}]
set_property PACKAGE_PIN W4 [get_ports {an[3]}]

set_property IOSTANDARD LVCMOS33 [get_ports {an[*]}]