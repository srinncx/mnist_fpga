// ============================================================
// TOP MODULE
// Basys-3 MNIST FPGA System
// UART → MNIST → 7-Segment Display
// ============================================================

module top (
    input  wire clk,        // 100 MHz clock
    input  wire rst,        // reset button
    input  wire rx,         // UART RX

    output wire [6:0] seg,  // 7-segment segments
    output wire [3:0] an    // 7-segment anodes
);

// ============================================================
// UART RECEIVER
// ============================================================

wire [7:0] uart_data;
wire       uart_valid;

uart_rx uart_inst (
    .clk(clk),
    .rst(rst),
    .rx(rx),
    .data_out(uart_data),
    .data_valid(uart_valid)
);

// ============================================================
// INPUT MEMORY LOADING (784 signed bytes)
// ============================================================

reg signed [7:0] input_mem [0:783];

reg [9:0] uart_counter;
reg       uart_done;

always @(posedge clk) begin
    if (rst) begin
        uart_counter <= 0;
        uart_done    <= 0;
    end
    else begin
        uart_done <= 0;

        if (uart_valid) begin
            input_mem[uart_counter] <= uart_data;
            uart_counter <= uart_counter + 1;

            if (uart_counter == 783) begin
                uart_done <= 1;     // trigger inference
                uart_counter <= 0;  // ready for next image
            end
        end
    end
end

// ============================================================
// MNIST INFERENCE MODULE
// ============================================================

wire [3:0] predicted_digit;
wire       inference_done;

mnist_inference inference_inst (
    .clk(clk),
    .rst(rst),
    .uart_done(uart_done),
    .predicted(predicted_digit),
    .done(inference_done)
);

// Connect top input memory to inference memory
// (Shared memory model for functional correctness)

integer i;
always @(posedge clk) begin
    for (i = 0; i < 784; i = i + 1) begin
        inference_inst.input_mem[i] <= input_mem[i];
    end
end

// ============================================================
// 7-SEGMENT DISPLAY (Single Digit)
// ============================================================

// Always enable first digit only
assign an = 4'b1110;

reg [6:0] seg_reg;
assign seg = seg_reg;

// Digit decoder
always @(*) begin
    case (predicted_digit)
        4'd0: seg_reg = 7'b1000000;
        4'd1: seg_reg = 7'b1111001;
        4'd2: seg_reg = 7'b0100100;
        4'd3: seg_reg = 7'b0110000;
        4'd4: seg_reg = 7'b0011001;
        4'd5: seg_reg = 7'b0010010;
        4'd6: seg_reg = 7'b0000010;
        4'd7: seg_reg = 7'b1111000;
        4'd8: seg_reg = 7'b0000000;
        4'd9: seg_reg = 7'b0010000;
        default: seg_reg = 7'b1111111;
    endcase
end

endmodule