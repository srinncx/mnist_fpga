// ============================================================
// MNIST Single-MAC Inference Engine
// Functional-Correct Version
// Signed INT8
// 1-cycle-per-pixel MAC
// Sequential ARGMAX
// ============================================================

module mnist_inference (
    input  wire        clk,          // 100 MHz
    input  wire        rst,          // synchronous reset
    input  wire        uart_done,    // asserted when UART finished loading input
    output reg  [3:0]  predicted,    // final predicted digit
    output reg         done          // high when inference complete
);

// ============================================================
// START PULSE GENERATION (Edge Detect)
// ============================================================

reg uart_done_d;
wire start;

always @(posedge clk) begin
    uart_done_d <= uart_done;
end

assign start = uart_done & ~uart_done_d;   // 1-cycle pulse

// ============================================================
// BRAM MEMORY
// ============================================================

reg signed [7:0] weights_mem [0:7839];
reg signed [7:0] bias_mem    [0:9];
reg signed [7:0] input_mem   [0:783];   // filled by UART

initial begin
    $readmemh("weights.mem", weights_mem);
    $readmemh("bias.mem",    bias_mem);
end

// ============================================================
// INTERNAL REGISTERS
// ============================================================

reg [12:0] weight_addr;     // 0–7839
reg [9:0]  pixel_addr;      // 0–783
reg [3:0]  neuron_index;    // 0–9

reg signed [7:0] weight_data;
reg signed [7:0] pixel_data;

reg signed [31:0] accumulator;
reg signed [31:0] result [0:9];

wire signed [15:0] product;
assign product = pixel_data * weight_data;

// ARGMAX REGISTERS
reg [3:0]  argmax_counter;
reg [3:0]  argmax_index;
reg signed [31:0] argmax_value;

// ============================================================
// FSM STATES
// ============================================================

localparam IDLE        = 3'd0,
           INIT_NEURON = 3'd1,
           MAC         = 3'd2,
           ADD_BIAS    = 3'd3,
           STORE       = 3'd4,
           NEXT_NEURON = 3'd5,
           ARGMAX_INIT = 3'd6,
           ARGMAX_RUN  = 3'd7;

reg [2:0] state;

// ============================================================
// BRAM READ (Synchronous)
// ============================================================

always @(posedge clk) begin
    weight_data <= weights_mem[weight_addr];
    pixel_data  <= input_mem[pixel_addr];
end

// ============================================================
// MAIN FSM
// ============================================================

always @(posedge clk) begin

    if (rst) begin
        state        <= IDLE;
        done         <= 0;
        predicted    <= 0;

        weight_addr  <= 0;
        pixel_addr   <= 0;
        neuron_index <= 0;
        accumulator  <= 0;
    end

    else begin
        case(state)

        // ----------------------------------------------------
        //IDLE
        // ----------------------------------------------------
        IDLE: begin
            done <= 0;
            if (start) begin
                neuron_index <= 0;
                weight_addr  <= 0;
                pixel_addr   <= 0;
                accumulator  <= 0;
                state <= INIT_NEURON;
            end
        end

        // ----------------------------------------------------
        // PREPARE NEW NEURON
        // ----------------------------------------------------
        INIT_NEURON: begin
            accumulator <= 0;
            pixel_addr  <= 0;
            state <= MAC;
        end

        // ----------------------------------------------------
        // MAC LOOP (1 cycle per pixel)
        // ----------------------------------------------------
        MAC: begin
            accumulator <= accumulator + product;

            weight_addr <= weight_addr + 1;
            pixel_addr  <= pixel_addr + 1;

            if (pixel_addr == 783) begin
                state <= ADD_BIAS;
            end
        end

        // ----------------------------------------------------
        // ADD BIAS
        // ----------------------------------------------------
        ADD_BIAS: begin
            accumulator <= accumulator + bias_mem[neuron_index];
            state <= STORE;
        end

        // ----------------------------------------------------
        // STORE RESULT
        // ----------------------------------------------------
        STORE: begin
            result[neuron_index] <= accumulator;
            state <= NEXT_NEURON;
        end

        // ----------------------------------------------------
        // MOVE TO NEXT NEURON
        // ----------------------------------------------------
        NEXT_NEURON: begin
            if (neuron_index == 9) begin
                state <= ARGMAX_INIT;
            end
            else begin
                neuron_index <= neuron_index + 1;
                state <= INIT_NEURON;
            end
        end

        // ----------------------------------------------------
        // ARGMAX INITIALIZE
        // ----------------------------------------------------
        ARGMAX_INIT: begin
            argmax_index   <= 0;
            argmax_value   <= result[0];
            argmax_counter <= 1;
            state <= ARGMAX_RUN;
        end

        // ----------------------------------------------------
        // ARGMAX RUN (10 cycles total)
        // ----------------------------------------------------
        ARGMAX_RUN: begin
            if (result[argmax_counter] > argmax_value) begin
                argmax_value <= result[argmax_counter];
                argmax_index <= argmax_counter;
            end

            if (argmax_counter == 9) begin
                predicted <= argmax_index;
                done <= 1;
                state <= IDLE;
            end
            else begin
                argmax_counter <= argmax_counter + 1;
            end
        end

        endcase
    end
end

endmodule