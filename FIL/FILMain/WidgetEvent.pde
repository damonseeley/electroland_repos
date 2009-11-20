public class WidgetEvent {
  
  // WIDGETEVENT.pde
  // holds properties on widget events to be passed to the manager.

  public Widget widget;			// reference to the control
  public String name;			// name of the control
  public int type;			// type of event
  public boolean state;			// state of the event
	
  public WidgetEvent(Widget widget, int type, boolean state){
    this.widget = widget;
    this.name = widget.name;
    this.type = type;
    this.state = state;
  }
	
}
