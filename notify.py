from gi.repository import Gtk

dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Average Reward: {}".format(avg_reward))
dialog.format_secondary_text(
    "Go checkout the value function. Close this window to resume learning.")
dialog.run()
