// ============================================================
// UART Receiver (8N1)
// Basys-3: 100 MHz Clock
// Baud: 115200
// ============================================================

module uart_rx #

(
    parameter CLKS_PER_BIT = 868   // 100MHz / 115200 ≈ 868
)

(
    input  wire clk,
    input  wire rst,
    input  wire rx,                // UART RX line

    output reg  [7:0] data_out,    // received byte
    output reg        data_valid   // 1-cycle pulse when byte ready
);

// ============================================================
// FSM STATES
// ============================================================

localparam IDLE        = 3'd0,
           START_BIT   = 3'd1,
           DATA_BITS   = 3'd2,
           STOP_BIT    = 3'd3,
           CLEANUP     = 3'd4;

reg [2:0] state;

// ============================================================
// INTERNAL REGISTERS
// ============================================================

reg [15:0] clk_count;
reg [2:0]  bit_index;
reg [7:0]  rx_shift;

// ============================================================
// MAIN FSM
// ============================================================

always @(posedge clk) begin

    if (rst) begin
        state      <= IDLE;
        clk_count  <= 0;
        bit_index  <= 0;
        data_out   <= 0;
        data_valid <= 0;
    end

    else begin
        case (state)

        // ----------------------------------------------------
        // WAIT FOR START BIT (rx goes LOW)
        // ----------------------------------------------------
        IDLE: begin
            data_valid <= 0;
            clk_count  <= 0;
            bit_index  <= 0;

            if (rx == 1'b0) begin     // Start bit detected
                state <= START_BIT;
            end
        end

        // ----------------------------------------------------
        // VERIFY START BIT (sample in middle)
        // ----------------------------------------------------
        START_BIT: begin
            if (clk_count == (CLKS_PER_BIT/2)) begin
                if (rx == 1'b0) begin
                    clk_count <= 0;
                    state <= DATA_BITS;
                end
                else begin
                    state <= IDLE;   // false start
                end
            end
            else begin
                clk_count <= clk_count + 1;
            end
        end

        // ----------------------------------------------------
        // READ 8 DATA BITS (LSB first)
        // ----------------------------------------------------
        DATA_BITS: begin
            if (clk_count == CLKS_PER_BIT - 1) begin
                clk_count <= 0;

                rx_shift[bit_index] <= rx;

                if (bit_index == 7) begin
                    bit_index <= 0;
                    state <= STOP_BIT;
                end
                else begin
                    bit_index <= bit_index + 1;
                end
            end
            else begin
                clk_count <= clk_count + 1;
            end
        end

        // ----------------------------------------------------
        // READ STOP BIT
        // ----------------------------------------------------
        STOP_BIT: begin
            if (clk_count == CLKS_PER_BIT - 1) begin
                data_out   <= rx_shift;
                data_valid <= 1;   // 1-cycle pulse
                clk_count  <= 0;
                state <= CLEANUP;
            end
            else begin
                clk_count <= clk_count + 1;
            end
        end

        // ----------------------------------------------------
        // CLEANUP
        // ----------------------------------------------------
        CLEANUP: begin
            data_valid <= 0;
            state <= IDLE;
        end

        endcase
    end
end

endmodule