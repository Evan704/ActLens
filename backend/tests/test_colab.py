from actlens.colab import find_tunnel_url


def test_find_tunnel_url():
    log = "INF |  Your quick Tunnel has been created! Visit it at:  |\nINF |  https://foo-bar-12.trycloudflare.com  |"
    assert find_tunnel_url(log) == "https://foo-bar-12.trycloudflare.com"
    assert find_tunnel_url("INF api.trycloudflare.com is not a tunnel url") is None
