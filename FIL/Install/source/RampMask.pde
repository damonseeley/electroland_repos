public class RampMask{
  
  // RAMPMASK.pde
  // used to mask off projection over the ramp area, as well as push textblocks away.
  
  private int x1, y1;  // top left
  private int x2, y2;  // rop right
  private int x3, y3;  // bottom right
  private int x4, y4;  // bottom left
  private int rampTopLeft[] = new int[2];     // top left
  private int rampTopRight[] = new int[2];    // top right
  private int rampBottomRight[] = new int[2];  // bottom right
  private int rampBottomLeft[] = new int[2];  // bottom left
  private int rampCenter[] = new int[2];       // ramp center
  
  public RampMask(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4){
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
    this.x3 = x3;
    this.y3 = y3;
    this.x4 = x4;
    this.y4 = y4;
    
    rampTopLeft[0] = x1;
    rampTopLeft[1] = y1;
    rampTopRight[0] = x2;
    rampTopRight[1] = y2;
    rampBottomRight[0] = x3;
    rampBottomRight[1] = y3;
    rampBottomLeft[0] = x4;
    rampBottomLeft[1] = y4;
    rampCenter[0] = x1 + (x2 - x1)/2;
    rampCenter[1] = y1 + (y3 - (y1 + ((y2 - y1)/2)))/2;  // compensate for ramp of top edge
  }
  
  public void checkCollisions(ConcurrentHashMap textBlocks){
    
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      /*
      // THIS ONLY AFFECTS BLOCKS WITH BOTTOM EDGE DIRECTLY INTERSECTING RAMP SLOPE
      // check bottom edge of text against ramp sloping edge
      int[] vec1 = new int[2];    // bottom left
      vec1[0] = int(b.getX() - b.getWidth()/2);
      vec1[1] = int(b.getY() + b.getHeight()/2);
      int[] vec2 = new int[2];    // bottom right
      vec2[0] = int(b.getX() + b.getWidth()/2);
      vec2[1] = int(b.getY() + b.getHeight()/2);
      if(edgeIntersection(vec1, vec2, rampVec1, rampVec2)){
        //println(b.getText() + " intersecting ramp");
        b.push(0, -1);  // arbitrary upward force
      }
      */
      
      // if the line from any of textblocks corners to the center of the ramp DOESN'T cross any of the ramps edges...
      // then it must be inside the ramp polygon, and the textblock needs to be pushed away
      int[] topleft = new int[2];
      topleft[0] = int(b.getX() - b.getWidth()/2);
      topleft[1] = int(b.getY() - b.getHeight()/2);
      int[] topright = new int[2];
      topright[0] = int(b.getX() + b.getWidth()/2);
      topright[1] = int(b.getY() - b.getHeight()/2);
      int[] bottomleft = new int[2];
      bottomleft[0] = int(b.getX() - b.getWidth()/2);
      bottomleft[1] = int(b.getY() + b.getHeight()/2);
      int[] bottomright = new int[2];
      bottomright[0] = int(b.getX() + b.getWidth()/2);
      bottomright[1] = int(b.getY() + b.getHeight()/2);
      boolean inside = false;
      
      // check topleft corner against all edges
      if(!edgeIntersection(topleft, rampCenter, rampTopLeft, rampTopRight) &&
         !edgeIntersection(topleft, rampCenter, rampTopRight, rampBottomRight) &&
         !edgeIntersection(topleft, rampCenter, rampBottomRight, rampBottomLeft) &&
         !edgeIntersection(topleft, rampCenter, rampTopLeft, rampBottomRight)){
         inside = true;
      }
      // check topright corner against all edges
      if(!edgeIntersection(topright, rampCenter, rampTopLeft, rampTopRight) &&
         !edgeIntersection(topright, rampCenter, rampTopRight, rampBottomRight) &&
         !edgeIntersection(topright, rampCenter, rampBottomRight, rampBottomLeft) &&
         !edgeIntersection(topright, rampCenter, rampTopLeft, rampBottomRight)){
         inside = true;
      }
      // check bottomleft corner against all edges
      if(!edgeIntersection(bottomleft, rampCenter, rampTopLeft, rampTopRight) &&
         !edgeIntersection(bottomleft, rampCenter, rampTopRight, rampBottomRight) &&
         !edgeIntersection(bottomleft, rampCenter, rampBottomRight, rampBottomLeft) &&
         !edgeIntersection(bottomleft, rampCenter, rampTopLeft, rampBottomRight)){
         inside = true;
      }
      // check bottomright corner against all edges
      if(!edgeIntersection(bottomright, rampCenter, rampTopLeft, rampTopRight) &&
         !edgeIntersection(bottomright, rampCenter, rampTopRight, rampBottomRight) &&
         !edgeIntersection(bottomright, rampCenter, rampBottomRight, rampBottomLeft) &&
         !edgeIntersection(bottomright, rampCenter, rampTopLeft, rampBottomRight)){
         inside = true;
      }
      
      if(inside){
        b.push(0,-1);
      }
      
    }
  }
  
  // this came from http://gpwiki.org/index.php/Polygon_Collision
  private double determinant(int[] vec1, int[] vec2){
    return vec1[0]*vec2[1]-vec1[1]*vec2[0];
  }
 
  //one edge is a-b, the other is c-d
  boolean edgeIntersection(int[] a, int[] b, int[] c, int[] d){
    double det=determinant(subtractVector(b,a),subtractVector(c,d));
    double t=determinant(subtractVector(c,a),subtractVector(c,d))/det;
    double u=determinant(subtractVector(b,a),subtractVector(c,a))/det;
    if ((t<0)||(u<0)||(t>1)||(u>1)){
      return false;
    }
    //return a*(1-t)+t*b;
    return true;
  }
  
  int[] subtractVector(int[] a, int[] b){
    int[] newvec = new int[2];
    newvec[0] = a[0] - b[0];
    newvec[1] = a[1] - b[1];
    return newvec;
  }
  
  public void draw(){  // only drawn on projection version
    noStroke();
    fill(0);
    quad(x1, y1, x2, y2, x3, y3, x4, y4);
  }
  
}
