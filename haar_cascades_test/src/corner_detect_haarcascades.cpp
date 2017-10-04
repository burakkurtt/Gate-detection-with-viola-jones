 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

 #include <iostream>
 #include <stdio.h>
 #include <time.h>
 #include <math.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );

 /** Global variables */
 // trained xml files for gate corners should put inside xml file and changed below names
 String lb_corner_name = "../xml/name of trained corner xml file";
 String rb_corner_name = "../xml/name of trained corner xml file";
 String rt_corner_name = "../xml/name of trained corner xml file";

 CascadeClassifier corner_cascade;
 CascadeClassifier rightbottom_corner_cascade;
 CascadeClassifier topright_corner_cascade;

 string window_name = "Capture - corner detection";
 RNG rng(12345);

 /** @function main */
 int main( int argc, const char** argv )
 {
   VideoCapture capture(-1);
   Mat frame;

   //-- 1. Load the cascades
   if( !corner_cascade.load( rb_corner_name ) ){ printf("--(!)Error loading - left bottom corner xml file loading error\n"); return -1; };
   if( !rightbottom_corner_cascade.load( rb_corner_name ) ){ printf("--(!)Error loading - right bottom corner xml file loading error\n"); return -1; };
   if( !topright_corner_cascade.load( rt_corner_name ) ){ printf("--(!)Error loading - right top corner xml file loading error\n"); return -1; };
   //-- 2. Read the video stream
   if( capture.isOpened())
   {
     while( true )
     {
      clock_t t;
      t = clock();
      capture.read(frame);  
      //-- 3. Apply the classifier to the frame
      if( !frame.empty() )
      { detectAndDisplay( frame ); }
      else
      { printf(" --(!) No captured frame -- Break!"); break; }
      t = clock() - t;
      cout << "time per loop" << ((float)t)/CLOCKS_PER_SEC << endl;
      int c = waitKey(10);
      if( (char)c == 'c' ) { break; }
      }
   }
   return 0;
 }
/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> corners;
  std::vector<Rect> corners_R;
  std::vector<Rect> corners_TR;
  Mat frame_gray;
  Mat midframe;
  Mat frame_gray_BR;
  Mat frame_gray_TR;
  //frame size
  int width = frame.size().width;
  int height = frame.size().height;
  // finding 1/3*frame with roi
  Rect roi; 
  roi.x = 0;
  roi.y = height/3;
  roi.width = width-1;
  roi.height = height/3; 
  if(roi.x >= 0 && roi.y >= 0 && roi.width + roi.x < frame.cols && roi.height + roi.y < frame.rows)
  {
    midframe = frame(roi);
    cvtColor( midframe, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect corners
    corner_cascade.detectMultiScale( frame_gray, corners, 1.05, 3, 0|CV_HAAR_SCALE_IMAGE, Size(10, 40) );
    for( size_t i = 0; i < corners.size(); i++ )
    {     
      //finding center of found corner
      Point center( corners[i].x + corners[i].width*0.5, corners[i].y + corners[i].height*0.5 );
      //drawing center of LB corners  
      rectangle( frame,
           Point( center.x - 10, (center.y + height/3) - 10),
           Point( center.x + 10, (center.y + height/3) + 10),
           Scalar( 0, 255, 255 ),
           1);    
      // finding interested area of midframe for bottom right corner  -------------------------------------------------------------------------
      int midwidth = midframe.size().width;
      int midheight = midframe.size().height;
      Rect roi_bottom;
      roi_bottom.x = center.x + 0.01 * midwidth;
      roi_bottom.y = center.y - 0.7 * midheight/3;
      roi_bottom.width = midwidth - roi_bottom.x - 0.01 * midwidth; 
      roi_bottom.height = 1.4 * midheight/3;
      if(roi_bottom.x >= 0 && roi_bottom.y >= 0 && roi_bottom.width + roi_bottom.x < midframe.cols && roi_bottom.height + roi_bottom.y < midframe.rows)
        {
          // drawing rectangular of area of interested
          rectangle( midframe,
              Point( roi_bottom.x, roi_bottom.y),
              Point( roi_bottom.x + roi_bottom.width , roi_bottom.y + roi_bottom.height),
              Scalar( 0, 0, 255 ),
              1);
          Mat rigthbottom_frame = midframe(roi_bottom);
          cvtColor( rigthbottom_frame, frame_gray, CV_BGR2GRAY );
          equalizeHist( frame_gray, frame_gray_BR );  
          // finding corners at interested area 
          rightbottom_corner_cascade.detectMultiScale( frame_gray_BR, corners_R, 1.05, 3, 0|CV_HAAR_SCALE_IMAGE, Size(10, 30) );
          for( size_t j = 0; j < corners_R.size(); j++ )
            {
              // finding center of BR corners
              Point center_R( roi_bottom.x + corners_R[j].x + corners_R[j].width*0.5, height/3 + roi_bottom.y + corners_R[j].y + corners_R[j].height*0.5 );
              rectangle( frame,
                  Point( center_R.x - 10, center_R.y - 10),
                  Point( center_R.x + 10, center_R.y + 10),
                  Scalar( 255, 255, 0 ),
                  1);
              // right top corner finding ------------------------------------------------------------------------------------------------------
              Rect roi_topright;
              roi_topright.x = center_R.x + 0.01 * width;
              roi_topright.y = center_R.y - height/6; //0.4 * height/3;
              roi_topright.width = width - center_R.x - 0.01 * width; 
              roi_topright.height = height/3; //0.8 * height/3; 
              if(roi_topright.x >= 0 && roi_topright.y >= 0 && roi_topright.width + roi_topright.x < frame.cols && roi_topright.height + roi_topright.y < frame.rows)
                {
                  Mat topright_frame = frame(roi_topright);
                  cvtColor( topright_frame, frame_gray, CV_BGR2GRAY );
                  equalizeHist( frame_gray, frame_gray_TR );  
                  // drawing rectangular of area of interested
                  rectangle( frame,
                      Point( roi_topright.x, roi_topright.y),
                      Point( roi_topright.x + roi_topright.width ,roi_topright.y + roi_topright.height),
                      Scalar( 0, 0, 255 ),
                      1);
                  // finding TR corners interested area 
                  topright_corner_cascade.detectMultiScale( frame_gray_TR, corners_TR, 1.05, 3, 0|CV_HAAR_SCALE_IMAGE, Size(10, 30) );
                  for( size_t k = 0; k < corners_TR.size(); k++ )
                    {
                      Point center_TR( roi_topright.x + corners_TR[k].x + corners_TR[k].width*0.5, roi_topright.y + corners_TR[k].y + corners_TR[k].height*0.5 );
                      rectangle( frame,
                          Point( center_TR.x - 10, center_TR.y - 10),
                          Point( center_TR.x + 10, center_TR.y + 10),
                          Scalar( 0, 255, 0 ),
                          1);
                      rectangle( frame,
                          Point( center.x, center_TR.y),
                          Point( center_R.x, center_R.y),
                          Scalar( 0, 255, 0 ),
                          3);
                      // writing findin corners location on frame
                      cout << "center BL " << center.x << " " << center.y + height/3 << endl;
                      cout << "center BR " << center_R.x << " " << center_R.y << endl;
                      cout << "center TL " << center_TR.x << " " << center_TR.y << endl;
                      break;
                    }
                } else {
                    cout << "top right frame extend the frame constrains" << endl;
                }
            break;
            }
        } else {
            cout << " bottom frame extend the frame constrains " << endl;  
        }   
      break;
    }
    //Show 1/3 of frame 
    line(frame,
           Point(0, height/3),
           Point(width, height/3),
           Scalar( 0, 0, 255 ),
           1);
    line(frame,
           Point(0, 2*height/3),
           Point(width, 2*height/3),
           Scalar( 0, 0, 255 ),
           1);
    imshow( window_name, frame );
  } else {  
    cout << "midframe extend the frame constrains" << endl;
  } 
 }