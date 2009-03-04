package net.electroland.enteractive.gui.widgets;

/**
 * Contains information on the type of event and where it came from.
 * @author asiegel
 */

public class WidgetEvent {

	public Widget widget;			// reference to the control
	public String name;			// name of the control
	public int type;				// type of event
	public boolean state;			// state of the event
	
	public WidgetEvent(Widget widget, int type, boolean state){
		this.widget = widget;
		this.name = widget.name;
		this.type = type;
		this.state = state;
	}
	
}
