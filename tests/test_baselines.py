from manet.simulator import MANETSimulator
from routing.aodv import AODVRouting
from routing.gpsr import GPSRRouting


def run_test():
    sim = MANETSimulator(num_nodes=30, seed=42)
    aodv = AODVRouting()
    gpsr = GPSRRouting()

    print("--- Testing AODV ---")
    sim.reset(seed=42)
    aodv.reset()
    for _ in range(50):
        sim.step_mobility()
        pkt = sim.generate_packet()
        if pkt:
            aodv.route_packet(pkt, sim)

    m_aodv = sim.get_metrics()
    print(f"AODV  PDR: {m_aodv['pdr']:.2f}%, Delay: {m_aodv['delay']*1000:.2f} ms, "
          f"Delivered: {int(m_aodv['total_delivered'])}/{int(m_aodv['total_generated'])}, "
          f"Link breaks: {int(m_aodv['link_breaks'])}")

    print("--- Testing GPSR ---")
    sim.reset(seed=42)
    gpsr.reset()
    for _ in range(50):
        sim.step_mobility()
        pkt = sim.generate_packet()
        if pkt:
            gpsr.route_packet(pkt, sim)

    m_gpsr = sim.get_metrics()
    print(f"GPSR  PDR: {m_gpsr['pdr']:.2f}%, Delay: {m_gpsr['delay']*1000:.2f} ms, "
          f"Delivered: {int(m_gpsr['total_delivered'])}/{int(m_gpsr['total_generated'])}, "
          f"Link breaks: {int(m_gpsr['link_breaks'])}")


if __name__ == "__main__":
    run_test()
