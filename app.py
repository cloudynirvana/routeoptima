from flask import Flask, request, jsonify
import os
import numpy as np
from scipy.integrate import odeint
from skopt import gp_minimize
import networkx as nx

app = Flask(__name__)

def build_network(routes, hub=None):
    G = nx.DiGraph()
    for route in routes:
        G.add_edge(route['origin'], route['dest'], route_id=route['id'])
    return G, hub

def simulate_demand(r, K, t, noise_factor=0.05, weather_factor=0.1, real_data=None):
    if real_data is not None:
        if len(real_data) != len(t):
            raise ValueError("Real data length must match time steps")
        return np.array(real_data)
    base_demand = odeint(lambda D, t: r * D * (1 - D / K), 100, t).flatten()
    demand = base_demand * (1 - 0.5 * np.random.binomial(1, weather_factor, len(t)))
    return np.maximum(demand + noise_factor * np.random.normal(0, base_demand / 5, len(t)), 0)

def schedule_flights(demand, aircraft, hub=None, is_hub_route=False, prev_flights=None, hub_routes_count=1):
    flights = np.zeros_like(demand)
    max_flights_per_day = aircraft * 24
    hub_capacity = 20 if hub == 'ADD' else 10
    hub_daily_limit = 480 if hub == 'ADD' else 240
    hub_route_limit = hub_daily_limit / max(1, hub_routes_count)
    for day in range(len(demand) // 100):
        idx = slice(day * 100, (day + 1) * 100)
        day_demand = demand[idx]
        day_flights = np.minimum(np.ceil(day_demand / 100).astype(int), aircraft)
        if is_hub_route and hub and prev_flights is not None:
            arrivals = prev_flights[idx]
            day_flights = np.where(arrivals > 0, day_flights * 1.2, day_flights)
        total_day_flights = min(max_flights_per_day, np.sum(day_flights))
        flights[idx] = day_flights * (total_day_flights / np.sum(day_flights + 1e-6))
        flights[idx] = np.floor(flights[idx]).astype(int)
        if hub and is_hub_route:
            flights[idx] = np.minimum(flights[idx], hub_capacity)
    if hub and is_hub_route:
        if flights.sum() > hub_route_limit:
            flights = flights * (hub_route_limit / flights.sum())
            flights = np.floor(flights).astype(int)
    return flights

def objective(params, t, aircraft, route, G, hub, noise=0.05, weather=0.1, real_data=None, hub_routes_count=1):
    r, K = params
    is_hub_route = hub and (route['origin'] == hub or route['dest'] == hub)
    demand = simulate_demand(r, K, t, noise, weather, real_data)
    prev_flights = None
    if route['origin'] == hub:
        for u, v, d in G.in_edges(route['origin'], data=True):
            if d['route_id'] in results:
                prev_flights = results[d['route_id']]['flights']
                break
    flights = schedule_flights(demand, aircraft, hub=hub, is_hub_route=is_hub_route, prev_flights=prev_flights, 
                              hub_routes_count=hub_routes_count)
    unmet = np.sum(np.maximum(0, demand - flights * 100)) * 2
    costs = np.sum(flights * (20 + 20 * (demand / np.max(demand + 1e-6))))
    reward = np.sum(flights * 100) * (12.0 if not is_hub_route else 10.0)
    if is_hub_route and route['dest'] == hub:
        transfer_pax = 0.2 * flights.sum() * 100
        reward += transfer_pax * 5
    penalty = 100 * aircraft * max(0, 0.75 * aircraft * 24 * 3 - flights.sum())
    demand_penalty = 200 * max(0, 400 - K) + 1000 * max(0, 0.1 - r)
    return unmet + costs - reward + penalty + demand_penalty

results = {}

def optimize_schedule(aircraft, horizon=72, routes=[{'id': 'R1'}], hub=None, noise=0.05, weather=0.1, real_data=None):
    global results
    t = np.linspace(0, horizon, int(horizon * 100 / 24))
    G, hub = build_network(routes, hub)
    hub_routes_count = sum(1 for r in routes if hub in [r['origin'], r['dest']])
    results = {}
    for route in routes:
        result = gp_minimize(
            lambda params: objective(params, t, aircraft, route, G, hub, noise, weather, real_data, hub_routes_count),
            [(0.05, 0.3), (400, 700)],
            n_calls=80, random_state=42, x0=[0.2, 600]
        )
        r, K = result.x
        is_hub_route = hub and (route['origin'] == hub or route['dest'] == hub)
        demand = simulate_demand(r, K, t, noise, weather, real_data)
        prev_flights = None
        if route['origin'] == hub:
            for u, v, d in G.in_edges(route['origin'], data=True):
                if d['route_id'] in results:
                    prev_flights = results[d['route_id']]['flights']
                    break
        flights = schedule_flights(demand, aircraft, hub=hub, is_hub_route=is_hub_route, prev_flights=prev_flights, 
                                  hub_routes_count=hub_routes_count)
        results[route['id']] = {
            'r': r, 'K': K, 'score': result.fun,
            'demand': demand, 'flights': flights,
            'total_pax': float(flights.sum() * 100),
            'cost': result.fun,
            'type': 'hub' if is_hub_route else 'direct'
        }
    output = {}
    for key, value in results.items():
        output[key] = {
            'r': value['r'], 'K': value['K'], 'score': value['score'],
            'demand': value['demand'].tolist(), 'flights': value['flights'].tolist(),
            'total_pax': value['total_pax'], 'cost': value['cost'], 'type': value['type']
        }
    return output

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    if 'real_data' in data:
        data['real_data'] = np.array(data['real_data'])
    result = optimize_schedule(**data)
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
