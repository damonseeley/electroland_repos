public class Slider extends Widget implements WidgetListener{
  
  private PImage img;
  public SliderBar sliderBar;
  
  public Slider(String name, int x, int y, float value, PImage img, PImage segmentImg, PImage segmentImgDown,
                PImage leftImg, PImage leftImgDown, PImage rightImg, PImage rightImgDown){
    super(name, x, y, img.width, img.height, value);
    this.img = img;
    sliderBar = new SliderBar("SliderBar",0,0,w,h,value, segmentImg, segmentImgDown, leftImg, leftImgDown, rightImg, rightImgDown);
    sliderBar.addListener(this);
  }
  
  public void dragged(){
    sliderBar.mouseDragged(mouseX-x, mouseY-y);
  }
  
  public void draw(){
    pushMatrix();
    translate(x,y,0);
    image(img,0,0);
    sliderBar.draw();
    popMatrix();
  }
  
  public void pressed(){
    sliderBar.mousePressed(mouseX-x, mouseY-y);
  }
  
  public void released(){
    sliderBar.mouseReleased(mouseX-x, mouseY-y);
  }
  
  public void rollOut(){
    // nothing?
  }
  
  public void rollOver(){
    // nothing?
  }
  
  public void cursorMovement(){
    sliderBar.mouseMoved(mouseX-x, mouseY-y);
  }
  
  public void setAreaVisible(float areaVisible){
    // normalized value used to shrink the width of the bar
    sliderBar.setWidth(int(w*areaVisible));
  }
  
  public void setOffset(float offset){
    // normalized value used to move the bar
    sliderBar.setOffset(int(w*offset));
  }
  
  public void widgetEvent(WidgetEvent we){
    value = we.widget.value;
    WidgetEvent newwe = new WidgetEvent(this, DRAGGED, true);
    super.newEvent(newwe);
  }
  
  public float getBarPosition(){
    //return sliderBar.getX() / (float)w;  // normalized location (left aligned)
    return (sliderBar.getX() + (sliderBar.getWidth()*0.5)) / (float)w;  // normalized location (centered)
  }
  
}
