#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "picojson.h"
#include <opencv2/opencv.hpp>



static void Matrixf_init(float *m, int row, int col)
{
  for(int i=0; i<row; i++){
    for(int j=0; j<col; j++){
      m[i*col + j] = 0;
    }
  }
}

static void Matrixf_set_identity(float *m, int row, int col)
{
  for(int i=0; i<row; i++){
    for(int j=0; j<col; j++){
      if(i == j){
        m[i*col + j] = 1;
      }
      else{
        m[i*col + j] = 0;
      }
    }
  }
}


static void Matrixf_assign(float *m1, float *m2, int row, int col)
{
  for(int i=0; i<row; i++){
    for(int j=0; j<col; j++){
      m1[i*col + j] = m2[i*col + j];
    }
  }
}


static void Matrixf_mul(float *m1, float *m2, float *r, 
                        int row1, int col1, int row2, int col2)
{
  for(int i=0; i<row1; i++){
    for(int j=0; j<col2; j++){
      r[i*col2 + j] = 0;
      for(int k=0; k<col1; k++){
        r[i*col2 + j] += m1[i*col1 + k] * m2[k*col2 + j];
      }
    }
  }
}


static void Matrixf_add(float *m1, float *m2, float *r, 
                        int row, int col)
{
  for(int i=0; i<row; i++){
    for(int j=0; j<col; j++){
        r[i*col + j] = m1[i*col + j] + m2[i*col + j];
    }
  }
}


static void Matrixf_mul_const(float *m1, float c, float *r, 
                              int row, int col)
{
  for(int i=0; i<row; i++){
    for(int j=0; j<col; j++){
      r[i*col + j] = c * m1[i*col + j];
    }
  }
}


static void Matrixf_sub(float *m1, float *m2, float *r, 
                        int row, int col)
{
  for(int i=0; i<row; i++){
    for(int j=0; j<col; j++){
        r[i*col + j] = m1[i*col + j] - m2[i*col + j];
    }
  }
}


static void Matrixf_T(float *m1, float *r,
                      int row, int col)
{
  for(int i=0; i<row; i++){
    for(int j=0; j<col; j++){
      r[j*row + i] = m1[i*col + j];
    }
  }
}


// tmp: float[col*2*row]
static void Matrixf_inv(float *m, float *r, int row, int col, float *tmp)
{
  for(int i=0; i<col; i++){

    for(int j=0; j<row; j++){
      tmp[i*(2*row) + j] = m[i*row + j];
    }

    for(int j=row; j<row*2; j++){
      if(j-row == i){
        tmp[i*(2*row) + j] = 1;
      }
      else{
        tmp[i*(2*row) + j] = 0;
      }
    }
  }

  // sweep(down)
  for(int i=0; i<col; i++){
    float pivot = tmp[i*(2*row) + i];
    int pivot_index = i;
    float pivot_tmp;
    for(int j=i; j<col; j++){
      if(tmp[j*(2*row) + i] > pivot){
        pivot = tmp[j*(2*row) + i];
        pivot_index = j;
      }
    }

    if(pivot_index != i){
      for(int j=0; j<2*row; j++){
        pivot_tmp = tmp[pivot_index * (2*row) + j];
        tmp[pivot_index * (2*row) + j] = tmp[i*(2*row)+j];
        tmp[i*(2*row) + j] = pivot_tmp;
      }
    }
    // division
    for(int j=0; j<2*row; j++){
      tmp[i*(2*row) + j] /= pivot;
    }

    // sweep
    for(int j=i+1; j<col; j++){
      float tmp2 = tmp[j*(2*row) + i];

      for(int k=0; k<row*2; k++){
        tmp[j*(2*row) + k] -= tmp2 * tmp[i*(2*row)+k];
      }
    }
  }

  // sweep up
  for(int i=0; i<col-1; i++){
    for(int j=i+1; j<col; j++){
      float pivot = tmp[(col-1-j)*(2*row) + (row-1-i)];
      for(int k=0; k<2*row; k++){
        tmp[(col-1-j)*(2*row)+k] -= pivot * tmp[(col-1-i)*(2*row)+k];
      }
    }
  }

  // copy
  for(int i=0; i<col; i++){
    for(int j=0; j<row; j++){
      r[i*row+j] = tmp[i*(2*row) + (j+row)];
    }
  }
}


static void Matrixf_print(float *m1,
                          int row, int col)
{
  for(int i=0; i<row; i++){
    for(int j=0; j<col; j++){
      printf("%.2f ", m1[i*col + j]);
    }
    printf("\n");
  }
}


#define True  (1)
#define False (0)

// Kalman Filter Constant Velocity Model
typedef struct {
  float variance[4][4];
  float measure_noise[2][2];
  float process_noise[4][4];
  float predict_state[4][1];
  float likelihood;

  float state_transition_matrix[4][4];
  // x = (x, y, vx, vy)
  // z = (x, y)
  // X = A * x
  // z = H * x
  float state[4][1];
  float measure_matrix[2][4];
  float kalman_gain[4][2];
  unsigned char inited_state;
} KFCV;



void KFCV_init(KFCV *kf)
{
  // set identity
  Matrixf_set_identity(kf->state_transition_matrix[0], 4, 4);
  Matrixf_init(kf->measure_matrix[0], 2, 4);
  kf->measure_matrix[0][0] = 1;
  kf->measure_matrix[1][1] = 1;
  Matrixf_set_identity(kf->variance[0], 4, 4);
  Matrixf_set_identity(kf->measure_noise[0], 2, 2);
  Matrixf_set_identity(kf->process_noise[0], 4, 4);

  kf->inited_state = False;
}

void KFCV_init_state(KFCV *kf, float x, float y)
{
  kf->state[0][0] = x;
  kf->state[1][0] = y;
  kf->state[2][0] = 0;
  kf->state[3][0] = 0;

  kf->inited_state = True;
}


void KFCV_get_state(KFCV *kf, float *x, float *y){
  *x = kf->state[0][0];
  *y = kf->state[1][0];
}


void KFCV_predict(KFCV *kf, float delta_t)
{
  if(kf->inited_state == False){
    return;
  }

  kf->state_transition_matrix[0][2] = delta_t;
  kf->state_transition_matrix[1][3] = delta_t;

  // [1, 0, tx, 0,
  //  0, 1, 0, ty,
  //  0, 0, 1, 0,
  //  0, 0, 0, 1] *
  // [x, y, vx, vy]
  
  // calc Next point
  // x = Ax + Bw 
  // x' = x + tx*vx
  float tmp0[4][1];
  Matrixf_mul(kf->state_transition_matrix[0], kf->state[0], 
              tmp0[0],
              4, 4,
              4, 1);
  Matrixf_assign(kf->state[0], tmp0[0], 4, 1);
  Matrixf_assign(kf->predict_state[0], tmp0[0], 4, 1);
  
  // variance
  // P' = AP-1At + BU-1Bt current process noise Bw = 0
  float tmp1[4][4];
  float Atrans[4][4];
  Matrixf_T(kf->state_transition_matrix[0], Atrans[0], 4, 4);
  Matrixf_mul(kf->state_transition_matrix[0], kf->variance[0],
              tmp1[0],
              4, 4,
              4, 4);
  Matrixf_mul(tmp1[0], Atrans[0],
              kf->variance[0],
              4, 4,
              4, 4);
  Matrixf_add(kf->variance[0], kf->process_noise[0],
              kf->variance[0],
              4, 4);

  //printf("variance\n");
  //Matrixf_print(kf->variance[0], 4, 4);
  
}


void KFCV_correct(KFCV *kf, float x, float y)
{
  if(kf->inited_state == False){
    KFCV_init_state(kf, x, y);
    return;
  }

  // x = A*x + B*u + noise1
  // z = C*x + noise2
  //
  // x = A*x + B*input
  // V : x 's cov
  // V = A*V*A^t + R1
  // K = V*C^t*(C*V*C^t+R2)^-1
  // x = x + K*(z - C*x)
  // V = (I - K*C)*V

  // measure cov: S = H*P*H^T + R
  //float measure[2][1];
  float cov1[2][4];
  float cov[2][2];  
  float measure_matrix_T[4][2];
  Matrixf_T(kf->measure_matrix[0], measure_matrix_T[0], 2, 4);

  Matrixf_mul(kf->measure_matrix[0], kf->variance[0],
              cov1[0],
              2, 4,
              4, 4);
  Matrixf_mul(cov1[0], measure_matrix_T[0],
              cov[0],
              2, 4,
              4, 2);
  Matrixf_add(cov[0], kf->measure_noise[0], cov[0], 2, 2);

  // Kalman gain
  // K = V * C^T * COV^-1
  float k[4][2];
  float cov_inv[2][2];
  // inv tmp[col * 2 * row];
  float inv_tmp[2*2*2];
  Matrixf_inv(cov[0], cov_inv[0], 2, 2, inv_tmp);

  Matrixf_mul(kf->variance[0], measure_matrix_T[0],
              k[0],
              4, 4,
              4, 2);
  Matrixf_mul(k[0], cov_inv[0],
              kf->kalman_gain[0],
              4, 2,
              2, 2);

  //float measure_matrix[2][4];
  //float kalman_gain[4][2];
  //float state[4][1];
  
  // variance
  float tmp1[4][4];
  float tmp2[4][4];
  Matrixf_mul(kf->kalman_gain[0], kf->measure_matrix[0],
              tmp1[0],
              4, 2,
              2, 4);
  Matrixf_mul(tmp1[0], kf->variance[0],
              tmp2[0],
              4, 4,
              4, 4);
  Matrixf_sub(kf->variance[0], tmp2[0],
              kf->variance[0],
              4, 4);

  /*
  float variance[4][4];
  float measure_noise[2][2];
  float process_noise[4][4];
  float predict_state[4][1];
  float likelihood;

  float state_transition_matrix[4][4];
  float state[4][1];
  float measure_matrix[2][4];
  float kalman_gain[4][2];
  */

  // state
  // state = state + K*(z - C*state)
  float tmp3[2][1];
  float tmp4[4][1];
  float measure[2][1];
  measure[0][0] = x;
  measure[1][0] = y;
  Matrixf_mul(kf->measure_matrix[0], kf->state[0],
              tmp3[0],
              2, 4,
              4, 1);
  Matrixf_sub(measure[0], tmp3[0], 
              measure[0],
              2, 1);
  Matrixf_mul(kf->kalman_gain[0], measure[0],
              tmp4[0],
              4, 2,
              2, 1);
  Matrixf_add(kf->state[0], tmp4[0], kf->state[0], 4, 1);

  //printf("measure\n");
  //Matrixf_print(measure[0], 2, 1);
  //printf("K\n");
  //Matrixf_print(kf->kalman_gain[0], 4, 1);
  

}



// low-pass filter
typedef struct {
  unsigned int inited;
  float alpha;
  float state[2];
} LPFilter;


void LPFilter_init(LPFilter *pf, float alpha)
{
  pf->alpha = alpha;
  pf->inited = False;
  pf->state[0] = 0.0f;
  pf->state[1] = 0.0f;
}

void LPFilter_get_state(LPFilter *pf, float *s1, float *s2)
{
  *s1 = pf->state[0];
  *s2 = pf->state[1];
}

void LPFilter_add(LPFilter *pf, float *z)
{
  if(pf->inited){
    pf->state[0] = z[0] + pf->alpha * (pf->state[0] - z[0]);
    pf->state[1] = z[1] + pf->alpha * (pf->state[1] - z[1]);
  }
  else{
    pf->state[0] = z[0];
    pf->state[1] = z[1];
    pf->inited = True;
  }
}


// Object template
//car H 1.45 W 1.65 L 4.5


typedef struct {
  float x;
  float y;
  float width;
  float height;
} RectF;

typedef struct {
  //unsigned int id;
  int type;
  double timestamp;
  RectF box;
  // L, W, H
  float size[3]; 
  float center[3];
} TrackObject;


unsigned int TARGET_UID = 0;

typedef struct {
  TrackObject latest_object;
  
  KFCV image_center; //predict
  LPFilter image_wh; //predict
  KFCV world_center;
  double start_time;
  unsigned int id;
  int type;
  unsigned int lost_age;
  unsigned int sweep;
} Target;


void RectF_center(RectF *box, float *x, float *y)
{
  *x = box->x + box->width / 2.0f;
  *y = box->y + box->height / 2.0f;
}


static void refine_box(RectF *box, int width, int height)
{
  if(box->x < 0){
    box->width += box->x;
    box->x = 0;
  }
  if(box->y < 0){
    box->height += box->y;
    box->y = 0;
  }
  if(box->x >= width){
    box->x = 0;
    box->width = 0;
  }
  if(box->y >= height){
    box->y = 0;
    box->height = 0;
  }
  box->width = box->x + box->width <= width ? box->width : width - box->x;
  box->height = box->y + box->height <= height ? box->height : height - box->y;
  if(box->width < 0){
    box->width = 0;
  }
  if(box->height < 0){
    box->height = 0;
  }
}

void Target_init(Target *target)
{
  target->id = TARGET_UID++;

  target->lost_age = 0;
  target->sweep = False;

  // For 2D
  KFCV_init(&target->image_center);

  float init_variance = 2500;
  float measure_variance = 60;
  float process_variance = 400;

  Matrixf_mul_const(target->image_center.variance[0], init_variance,
                    target->image_center.variance[0],
                    4, 4);
  Matrixf_mul_const(target->image_center.measure_noise[0], measure_variance,
                    target->image_center.measure_noise[0],
                    2, 2);
  Matrixf_mul_const(target->image_center.process_noise[0], process_variance,
                    target->image_center.process_noise[0],
                    4, 4);

  float alpha = 0.6f;
  LPFilter_init(&target->image_wh, alpha);

  // For 3D
  init_variance = 5.0f;
  measure_variance = 1.0f;
  process_variance = 1.0f;
  KFCV_init(&target->world_center);
  Matrixf_mul_const(target->world_center.variance[0], init_variance,
                    target->world_center.variance[0],
                    4, 4);
  Matrixf_mul_const(target->world_center.measure_noise[0], measure_variance,
                    target->world_center.measure_noise[0],
                    2, 2);
  Matrixf_mul_const(target->world_center.process_noise[0], process_variance,
                    target->world_center.process_noise[0],
                    4, 4);
}


void Target_add(Target *target, TrackObject *object)
{
  target->latest_object = *object;
  target->type = object->type;
  target->lost_age = 0;
}


int Target_is_lost(Target *target)
{
  if(target->lost_age > 0){
    return True;
  }

  return False;
}


static void Target_clear(Target *target)
{
  target->sweep = True;
}


void Target_predict(Target *target, double timestamp)
{
  double delta_t = timestamp - target->latest_object.timestamp;

  // For 2D
  KFCV_predict(&target->image_center, delta_t);
  
  // For 3D
  double delta_t_2 = delta_t * delta_t;


  float process_variance = 4.0f;

  // x = 1/2 * a * t^2 + v0 * t
  // v = a * t + v0
  float acc_variance = process_variance;
  float pos_variance = 0.25f * acc_variance * delta_t_2;// * delta_t_2;
  std::cout << "pos_v:" << pos_variance << std::endl;
  float vel_variance = acc_variance * delta_t_2;
  target->world_center.process_noise[0][0] = pos_variance;
  target->world_center.process_noise[1][1] = pos_variance;
  target->world_center.process_noise[2][2] = vel_variance;
  target->world_center.process_noise[3][3] = vel_variance;
  KFCV_predict(&target->world_center, delta_t);
}


void Target_update2d(Target *target, int image_width, int image_height)
{
  if(Target_is_lost(target) == True){
    return;
  }

  TrackObject *object = &target->latest_object;
  
  float center_x, center_y;
  RectF_center(&object->box, &center_x, &center_y);

  float wh[2] = {object->box.width, object->box.height};
  LPFilter_add(&target->image_wh, wh);

  KFCV_correct(&target->image_center, center_x, center_y);

  RectF box;
  LPFilter_get_state(&target->image_wh, &box.width, &box.height);

  float x, y;
  KFCV_get_state(&target->image_center, &x, &y);
  box.x = x - box.width / 2;
  box.y = y - box.height / 2;

  refine_box(&box, image_width, image_height);
  object->box = box;
}

void Target_update3d(Target *target)
{
  TrackObject *object = &target->latest_object;

  if(Target_is_lost(target) != True){
    // TODO direction
    float z[2] = {object->center[0], object->center[1]};
    // todo UNKNOWN_UNMOVABLE

    // measure noise
    float obj_car_x = object->center[0];
    float obj_car_z = object->center[2];
    float obj_distance = obj_car_x * obj_car_x + obj_car_z * obj_car_z;
    float measure_variance = 0.01f;
    float dist_err = obj_distance * measure_variance;
    Matrixf_set_identity(target->world_center.measure_noise[0], 2, 2);
    Matrixf_mul_const(target->world_center.measure_noise[0], dist_err,
                      target->world_center.measure_noise[0],
                      2, 2);

    KFCV_correct(&target->world_center, z[0], z[1]);

    float x, y;
    KFCV_get_state(&target->world_center, &x, &y);
    object->center[0] = x;
    object->center[1] = y;
  }
}

#define TARGET_RESERVE_AGE (6)

#define ObstacleTrackerCapacity (128)

typedef struct {
  unsigned int target;
  unsigned int object;
  float score;
} Hypothesis;


typedef struct {
  Target targets[ObstacleTrackerCapacity];
  int targets_num;
  unsigned int used[ObstacleTrackerCapacity];
  unsigned int used_num;

} ObstacleTracker;


void ObstacleTracker_init(ObstacleTracker *ot)
{
  ot->targets_num = 0;
}

static void clear_used(ObstacleTracker *ot, unsigned int used_num)
{
  for(int i=0; i<ObstacleTrackerCapacity; i++){
    ot->used[i] = False;
  }

  ot->used_num = used_num;
}


float gaussian(float x, float mu, float sigma) {
  return expf(-(x - mu) * (x - mu) / (2 * sigma * sigma));
}

static float score_motion(ObstacleTracker *ot,
                          Target *target,
                          TrackObject *object)
{
  float target_x, target_y;
  KFCV_get_state(&target->image_center, &target_x, &target_y);
  float object_x, object_y;
  RectF_center(&object->box, &object_x, &object_y);
  float width = object->box.width;
  float height = object->box.height;
  //printf("motion %f %f\n", target_x, object_x);
  float s = gaussian(object_x, target_x, width) * gaussian(object_y, target_y, height);
  return s;
}


static float score_shape(ObstacleTracker *ot,
                         Target *target,
                         TrackObject *object)
{
  float target_width, target_height;
  LPFilter_get_state(&target->image_wh, &target_width, &target_height);
  float width = object->box.width;
  float height = object->box.height;
  float s = (target_width - width) * (target_height - height) / (target_width * target_height);
  return -fabsf(s);
}


RectF RectF_intersect(RectF *rect1, RectF *rect2)
{
  float rect1_area = rect1->width * rect1->height;
  float rect2_area = rect2->width * rect2->height;

  float xmin = fmaxf(rect1->x, rect2->x);
  float xmax = fminf(rect1->x+rect1->width, rect2->x+rect2->width);
  float ymin = fmaxf(rect1->y, rect2->y);
  float ymax = fminf(rect1->y+rect1->height, rect2->y+rect2->height);

  RectF rect;
  if(xmin >= xmax || ymin >= ymax){
    rect.x = 0;
    rect.y = 0;
    rect.width = 0;
    rect.height = 0;

  }
  else{
    rect.x = xmin;
    rect.y = ymin;
    rect.width = xmax - xmin;
    rect.height = ymax - ymin;
  }

  return rect;
}

typedef struct {
  int width;   // image width 
  int height;  // image height
  float k_mat[9];
} ObjMapper;



void IBackProjectCanonical(const float *x, const float *K, float depth, float *X)
{
  X[0] = (x[0] - K[2]) * depth * (1.0 / K[0]);
  X[1] = (x[1] - K[5]) * depth * (1.0 / K[4]);
  X[2] = depth;
}


void ObjMapper_init(ObjMapper *m, float *K, int image_width, int image_height)
{
  m->width = image_width;
  m->height = image_height;
  for(int i=0; i<9; i++){
    m->k_mat[i] = K[i];
  }
}


static int solve_center_from_nearest_vertical(ObjMapper *m,
                                              const float *bbox, 
                                              const float *hwl, 
                                              /* float ry, */ 
                                              float *center, 
                                              float *center_2d)
{
  center[0] = center[1] = center[2] = 0.0f;
  float height_bbox = bbox[3] - bbox[1];
  float width_bbox = bbox[2] - bbox[0];


  float f = (m->k_mat[0] + m->k_mat[4]) / 2;
  // x = (f * X/Z)
  float depth = f * hwl[0] * (1.0 / height_bbox);

  // TODO
  //float PI = (float)M_PI;


  // back-project to solve center
  center_2d[0] = (bbox[0] + bbox[2]) / 2;
  center_2d[1] = (bbox[1] + bbox[3]) / 2;

  // TODO
  //get_center(bbox, depth, hwl, center, center_2d);
  IBackProjectCanonical(center_2d, m->k_mat, depth, center);

  return 0;
}


void ObjMapper_solve3dBBox(ObjMapper *m, float center[3], float hwl[3], float bbox[4])
{
  const float PI = (float)M_PI;
  const float PI_HALF = PI / 2;
  
  // angle

  float center_2d[2] = {0};

  solve_center_from_nearest_vertical(m, bbox, hwl, center, center_2d);

  //  std::cout << center[0] << "," << center[1] << "," << center[2] << std::endl;
}


void ObjMapper_transform(ObjMapper *m, Target *objects, int object_num)
{
  for(int k=0; k<object_num; k++){
    TrackObject *obj = &objects[k].latest_object;
    float center[3] = {0};
    float hwl[3] = {obj->size[1], obj->size[2], obj->size[0]};
    float bbox[4] = {obj->box.x, obj->box.y, obj->box.x + obj->box.width, obj->box.y + obj->box.height};

    ObjMapper_solve3dBBox(m, center, hwl, bbox);

    obj->center[0] = center[0];
    obj->center[1] = center[1] - obj->size[1]/2.0f;
    obj->center[2] = center[2];
  }
}


float calculate_IoU(RectF *rect1, RectF *rect2)
{
  float rect1_area = rect1->width * rect1->height;
  float rect2_area = rect2->width * rect2->height;

  float sum_area = rect1_area + rect2_area;

  float xmin = fmaxf(rect1->x, rect2->x);
  float xmax = fminf(rect1->x+rect1->width, rect2->x+rect2->width);
  float ymin = fmaxf(rect1->y, rect2->y);
  float ymax = fminf(rect1->y+rect1->height, rect2->y+rect2->height);

  if(xmin >= xmax || ymin >= ymax){
    return 0.0f;
  }
  else{
    float intersect = (xmax - xmin) * (ymax - ymin);
    return intersect / (sum_area - intersect);
  }
}

static float score_overlap(ObstacleTracker *ot,
                           Target *target,
                           TrackObject *object)
{
  RectF box_target;
  RectF box_object;

  KFCV_get_state(&target->image_center, &box_target.x, &box_target.y);
  LPFilter_get_state(&target->image_wh, &box_target.width, &box_target.height);
  
  box_target.x = box_target.x - box_target.width * 0.5f;
  box_target.y = box_target.y - box_target.height * 0.5f;

  box_object = object->box;

  float s = calculate_IoU(&box_target, &box_object);
  return s;
}


float kTypeAssociatedCost[3][3] = {
  // car,bus,ped
  {0.0, 0.2, 1.0},
  {0.2, 0.0, 1.0},
  {1.0, 1.0, 0.0},
};

static void generate_hypothesis(ObstacleTracker *ot, 
                                TrackObject *objects[], 
                                unsigned int objects_num)
{
  const int ScoreListCapacity = ObstacleTrackerCapacity * 4;
  Hypothesis score_list[ScoreListCapacity];

  unsigned int score_list_num = 0;
  for(int i=0; i<ot->targets_num; i++){
    for(int j=0; j<objects_num; j++){
      if(score_list_num >= ScoreListCapacity){
        continue;
      }

      Hypothesis *hypo = &score_list[score_list_num];
      hypo->target = i;
      hypo->object = j;
      // TODO appearance
      float sa = 0.0;
      float sm = score_motion(ot, &ot->targets[i], objects[j]);
      float ss = score_shape(ot, &ot->targets[i], objects[j]);
      float so = score_overlap(ot, &ot->targets[i], objects[j]);

      //printf("sm %f, ss %f, so %f\n", sm, ss, so);

      float weight_appearance = 1.0;
      float weight_motion = 0.5; //1.0;
      float weight_shape = 0.15; //1.0;
      float weight_overlap = 0.35; //1.0;

      hypo->score = weight_appearance * sa 
        + weight_motion * sm
        + weight_shape * ss
        + weight_overlap * so;
      
      hypo->score += -kTypeAssociatedCost[ot->targets[i].type][objects[j]->type];

      float target_threshold = 0.6;
      if(sm < 0.045 || hypo->score < target_threshold){
        continue;
      }
      else{
        score_list_num += 1;
      }
    }
  }

  if(score_list_num == 0){
    return;
  }

  // sort score_list
  for(int i=0; i<score_list_num; i++){
    for(int j=i+1; j<score_list_num; j++){
      if(score_list[i].score < score_list[j].score){
        Hypothesis tmp = score_list[i];
        score_list[i] = score_list[j];
        score_list[j] = tmp;
      }
    }
  }

  //printf("score_list_num %d\n", score_list_num);

  unsigned int used_target[ObstacleTrackerCapacity];
  for(int i=0; i<ObstacleTrackerCapacity; i++){
    used_target[i] = False;
  }

  for(int i=0; i<score_list_num; i++){
    Hypothesis *pair = &score_list[i];
    if(used_target[pair->target] || ot->used[pair->object]){
      continue;
    }

    //    printf("Hypo: match pair target:%d,%d object:%d\n", pair->target, ot->targets[pair->target].id, pair->object);

    Target *target = &ot->targets[pair->target];
    Target_add(target, objects[pair->object]);
    ot->used[pair->object] = True;
    used_target[pair->target] = True;
  }
}



void ObstacleTracker_predict(ObstacleTracker *ot, double timestamp)
{
  for(int i=0; i<ot->targets_num; i++){
    Target_predict(&ot->targets[i], timestamp);
  }
}


static int is_covered(RectF *rect1, RectF *rect2, float threshold)
{
  RectF inter = RectF_intersect(rect1, rect2);
  return (inter.width * inter.height) / (rect1->width * rect1->height) > threshold;
}

static int create_new_target(ObstacleTracker *ot, TrackObject *objects[], int objects_num)
{
  // TODO Some
  
  int created_count = 0;
  for(int i=0; i<objects_num; i++){
    if(ot->used[i] == True){
      continue;
    }

    int covered = False;
    for(int j=0; j<ot->targets_num; j++){
      if(is_covered(&objects[i]->box, &ot->targets[j].latest_object.box, 0.4f) == True &&
         objects[i]->type == ot->targets[j].type){
        covered = True;
        break;
      }
    }
    if(covered == True){
      continue;
    }

    if(ot->targets_num < ObstacleTrackerCapacity){
      Target *target = &ot->targets[ot->targets_num];
      ot->targets_num += 1;
      Target_init(target);
      Target_add(target, objects[i]);
      created_count += 1;
    }
  }

  return created_count;
}


static void clear_targets(ObstacleTracker *ot)
{
  int left = 0;
  int end = ot->targets_num-1;

  while(left <= end){
    if(ot->targets[left].sweep == True){
      while((left < end) && ot->targets[end].sweep == True){
        --end;
      }
      if(left >= end){
        break;
      }
      ot->targets[left] = ot->targets[end];
      --end;
    }
    left++;
  }

  ot->targets_num = left;
}


void ObstacleTracker_associate2d(ObstacleTracker *ot, 
                                 TrackObject *detected_objects, int detected_objects_num,
                                 int image_width, int image_height)
{
  // TODO similar

  // TODO Remove old
  for(int i=0; i<ot->targets_num; i++){
    ot->targets[i].lost_age += 1;
  }


  TrackObject *track_objects[ObstacleTrackerCapacity];
  unsigned int track_objects_num = 0;
  for(int i=0; i<detected_objects_num; i++){
    if(track_objects_num >= ObstacleTrackerCapacity){
      break;
    }
    
    track_objects[track_objects_num] = &detected_objects[i];
    track_objects_num += 1;
  }

  // TODO reference Correct size

  clear_used(ot, detected_objects_num);

  generate_hypothesis(ot, track_objects, track_objects_num);

  int new_count = create_new_target(ot, track_objects, track_objects_num);
  printf("new_count: %d, target_num %d\n", new_count, ot->targets_num);

  for(int i=0; i<ot->targets_num; i++){
    if(ot->targets[i].lost_age > TARGET_RESERVE_AGE){
      printf("lost\n");
      // TODO
      Target_clear(&ot->targets[i]);
    }
    else{
      //TODO update type
      //Target_update_type(&ot->targets[i]);
      Target_update2d(&ot->targets[i], image_width, image_height);
    }
  }

  clear_targets(ot);

  // TODO
  /*
  for(int i=0; i<ot->targets_num; i++){
    printf("Update2d center: %f %f,  %f %f\n", 
           ot->targets[i].latest_object.box.x + ot->targets[i].latest_object.box.width/2, 
           ot->targets[i].latest_object.box.y + ot->targets[i].latest_object.box.height/2, 
           ot->targets[i].latest_object.box.width, ot->targets[i].latest_object.box.height);
  }
  */

}


void ObstacleTracker_associate3d(ObstacleTracker *ot)
{
  for(int i=0; i<ot->targets_num; i++){
    Target_update3d(&ot->targets[i]);
  }
}



int main()
{
  double timestamp = 0;

  ObstacleTracker tracker;
  ObstacleTracker_init(&tracker);

  TrackObject objects[256];

  // json
  std::ifstream fs;
  picojson::value v;
  fs.open("output_5500.json", std::ios::binary);
  fs >> v;
  fs.close();
  picojson::object &obj = v.get<picojson::object>();
  std::cout << obj.size() << std::endl;

  // video 
  cv::VideoCapture cap("2.mp4");
  cap.set(cv::CAP_PROP_POS_FRAMES, 5500);

  for(int i=0; i<obj.size(); i++){
    cv::Mat frame;
    cap >> frame;

    int image_width = frame.cols;
    int image_height = frame.rows;
    std::ostringstream number;
    number << i;
    picojson::array& boxes = obj[number.str()].get<picojson::array>();

    for(int j=0; j<boxes.size(); j++){
      picojson::array &box = boxes[j].get<picojson::array>();
      //std::cout << box[1].get<double>() << std::endl;
      //std::cout << box[0].get<double>() << std::endl;
      //std::cout << box[3].get<double>() << std::endl;
      //std::cout << box[2].get<double>() << std::endl;

      objects[j].timestamp = timestamp;
      objects[j].box.x = int(box[1].get<double>());
      objects[j].box.y = int(box[0].get<double>());
      objects[j].box.width = int(box[3].get<double>() - objects[j].box.x);
      objects[j].box.height = int(box[2].get<double>() - objects[j].box.y);
      int type = int(box[4].get<double>());
      switch(type) {
      case 5:
        type = 1;
        break;
      case 6:
        type = 0;
        break;
      case 14:
        type = 2;
        break;
      default:
        type = 0;
      }
      objects[j].type = type; // 0:car, 1:bus, 2:person
      objects[j].size[0] = 4.5;  // L //car H 1.45 W 1.65 L 4.5
      objects[j].size[1] = 1.45; // H
      objects[j].size[2] = 1.65; // W
      cv::rectangle(frame, cv::Point(int(objects[j].box.x), int(objects[j].box.y)), 
                    cv::Point(int(box[3].get<double>()), int(box[2].get<double>())), cv::Scalar(0,0,200), 2, 1);
    }

    ObstacleTracker_predict(&tracker, timestamp);
    timestamp += 0.033;

    ObstacleTracker_associate2d(&tracker, objects, boxes.size(), image_width, image_height);

    for(int k=0; k<tracker.targets_num; k++){

      float cx, cy;
      KFCV_get_state(&tracker.targets[k].image_center, &cx, &cy);
      float w, h; 
      LPFilter_get_state(&tracker.targets[k].image_wh, &w, &h);
      int x1 = int(cx - w/2);
      int y1 = int(cy - h/2);
      int x2 = x1 + w;
      int y2 = y1 + h;
      // bbox
      cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,200,200), 1, 1);
      // ID, type
      std::ostringstream text;
      text << tracker.targets[k].id << ": type" << tracker.targets[k].type;
      cv::putText(frame,text.str(), cv::Point(int(x1), int(y1)), cv::FONT_HERSHEY_SIMPLEX, 
                  0.5, cv::Scalar(0,255,0), 1);
    }




    ObjMapper obj_mapper;
    float fx = (float)image_width/2.0;
    float fy = fx;
    float cx = (float)image_width/2.0;
    float cy = (float)image_height/2.0;
    float k_mat[9] = {
      fx, 0, cx,
      0, fy, cy,
      0, 0, 1.0
    };
    ObjMapper_init(&obj_mapper, k_mat, image_width, image_height);
    ObjMapper_transform(&obj_mapper, tracker.targets, tracker.targets_num);
    
    ObstacleTracker_associate3d(&tracker);

    for(int k=0; k<tracker.targets_num; k++){
      TrackObject *obj = &tracker.targets[k].latest_object;
      std::ostringstream text;
      text << std::fixed << std::setprecision(1) << obj->center[0] << ","
           << std::fixed << std::setprecision(1) << obj->center[1] << ","
           << std::fixed << std::setprecision(1) << obj->center[2];
      cv::putText(frame, text.str(), cv::Point(int(obj->box.x), int(obj->box.y+obj->box.height/2)), cv::FONT_HERSHEY_SIMPLEX, 
                  0.5, cv::Scalar(0,255,0), 1);
    }

    cv::imshow("img", frame);
    cv::waitKey(1);
  }
  return 0;
}
