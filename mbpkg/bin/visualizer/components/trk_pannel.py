import dash_mantine_components as dmc

def make_trk_pannel() -> dmc.Card:
    return dmc.Card([
        dmc.CardSection("Tracking settings"),
        dmc.Text("tracking settings")
    ])


