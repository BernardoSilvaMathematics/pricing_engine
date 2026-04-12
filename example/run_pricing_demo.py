from numerical_methods.pricingengine import BlackScholesEngine

engine = BlackScholesEngine(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)

bs = engine.bs_call()
pde = engine.pde_call()
mc, se = engine.mc_call()

print("BS:", bs)
print("PDE:", pde)
print("MC:", mc, "±", 1.96 * se)
print("Delta:", engine.bs_delta())





