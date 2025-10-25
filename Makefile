# Create logs directory
logs:
	mkdir -p logs

# Create custom configs
configs/realbasicvsr_x4_sharp.py: configs/realbasicvsr_x4.py
	@echo "Creating sharp config (dynamic_refine_thres=5)..."
	@sed 's/generator=dict(type='\''RealBasicVSRNet'\'', is_sequential_cleaning=True)/generator=dict(type='\''RealBasicVSRNet'\'', is_sequential_cleaning=True, dynamic_refine_thres=5)/' configs/realbasicvsr_x4.py > configs/realbasicvsr_x4_sharp.py

configs/realbasicvsr_x4_verysharp.py: configs/realbasicvsr_x4.py
	@echo "Creating very sharp config (dynamic_refine_thres=3, no sequential cleaning)..."
	@sed 's/generator=dict(type='\''RealBasicVSRNet'\'', is_sequential_cleaning=True)/generator=dict(type='\''RealBasicVSRNet'\'', is_sequential_cleaning=False, dynamic_refine_thres=3)/' configs/realbasicvsr_x4.py > configs/realbasicvsr_x4_verysharp.py

# Test with sharper output (dynamic_refine_thres=5)
test_sharp: logs configs/realbasicvsr_x4_sharp.py
	python inference_realbasicvsr_streaming.py \
		configs/realbasicvsr_x4_sharp.py \
		checkpoints/RealBasicVSR_x4.pth \
		data/xix2_short.mp4 \
		output_sharp.mp4 \
		--max_seq_len 10 \
		--num_workers 0 \
		--fps 30 \
		--benchmark \
		--log_file logs/sharp.log

# Test with very sharp output (dynamic_refine_thres=3, no sequential cleaning)
test_verysharp: logs configs/realbasicvsr_x4_verysharp.py
	python inference_realbasicvsr_streaming.py \
		configs/realbasicvsr_x4_verysharp.py \
		checkpoints/RealBasicVSR_x4.pth \
		data/xix2_short.mp4 \
		output_verysharp.mp4 \
		--max_seq_len 10 \
		--num_workers 0 \
		--fps 30 \
		--benchmark \
		--log_file logs/verysharp.log

# Compare smoothness: original vs sharp vs very sharp
test_compare_smoothness: test_optimal test_sharp test_verysharp
	@echo "Generated 3 videos for comparison:"
	@echo "  output_optimal.mp4 - Original (smooth, default settings)"
	@echo "  output_sharp.mp4 - Sharp (dynamic_refine_thres=5)"
	@echo "  output_verysharp.mp4 - Very sharp (thres=3, no sequential cleaning)"

# Clean outputs and logs
clean:
	rm -rf output_*.mp4 logs/*.log configs/realbasicvsr_x4_sharp.py configs/realbasicvsr_x4_verysharp.py

# Clean everything
clean_all:
	rm -rf output_*.mp4 logs/ configs/realbasicvsr_x4_sharp.py configs/realbasicvsr_x4_verysharp.py

.PHONY: logs test_optimal test_sharp test_verysharp test_compare_smoothness test_seq100 process_all clean clean_all
