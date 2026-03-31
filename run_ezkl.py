import ezkl

print("1. Generating settings...")
ezkl.gen_settings('dncnn_15.onnx', 'settings.json')
print('✅ settings.json created')

print("2. Compiling model...")
ezkl.compile_model('dncnn_15.onnx', 'compiled_model.ezkl', 'settings.json')
print('✅ compiled_model.ezkl created')

print("3. Setup (generating keys)...")
ezkl.setup('compiled_model.ezkl', 'vk.key', 'pk.key')
print('✅ keys generated')

print("4. Generating witness and proof...")
ezkl.gen_witness('compiled_model.ezkl', 'input.json', 'witness.json')
ezkl.prove('witness.json', 'compiled_model.ezkl', 'pk.key', 'proof.pf', 'settings.json')
print('✅ PROOF GENERATED! File: proof.pf')

print("5. Verifying proof...")
ezkl.verify('proof.pf', 'settings.json', 'vk.key')
print('✅ PROOF VERIFIED ✓')
