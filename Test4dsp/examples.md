# 6678上DMA搬运相关代码
这一部分已经包装成同步dma搬运函数dma_trans了（在最后）
```c
/**
********************************************************************************
* @file       M6678E_DMA.c
* @brief      实现M6678E DMA各种功能的接口函数。
* @version    V1.0
* @date       2021.05.19
* @attention
* \par
* ==============================================================================
* @n   (C) 飞腾DSP技术支持团队.
********************************************************************************
*/
#include <c6x.h>
#include "DMA.h"

/** @addtogroup DMA_FUNCTION
 @{ */

/**
********************************************************************************
 *   @n@b DMA_ParaConfig(volatile unsigned int DMA_Numb,volatile unsigned int Numb ,volatile unsigned int QueueNUM,volatile unsigned int SourceAddr,volatile unsigned int Acount,volatile unsigned int Bcount,volatile unsigned int DestinationAddr,volatile unsigned int IntEnable)
 *
 *   @b 功能：
 *   @n DMA配置函数
 *
 *   @b 形参：
 *   @verbatim
 *       DMA_Numb 为DMA部件的选择，为DMA0和DMA2其中的一个
 *       Numb  为通道号
 *       QueueNUM 为队列号   因为只有4个队列，所以该参数仅可配0~3 四个参数。
 *       SourceAddr为源地址
 *       Acount为数据字节数
 *       Bcount为数据帧数
 *       DestinationAddr为目的地址
 *       IntEnable为中断使能
 *   @endverbatim
 *
 *   <b> 返回值： </b>
 *   @n 无
 *
 *   <b> 注意事项： </b>
 *   @n 无
 *******************************************************************************
 */
void DMA_ParaConfig(volatile unsigned int DMA_Numb,volatile unsigned int Numb ,volatile unsigned int QueueNUM,volatile unsigned int SourceAddr,volatile unsigned int Acount,volatile unsigned int Bcount,volatile unsigned int DestinationAddr,volatile unsigned int IntEnable)
{
    volatile unsigned int DMA_BASE=0;
    volatile int  set_OPT_addr ;
    volatile int  DNUM_Temp=0;
    int  Numb_temp = 0 ;
    int  Bcount_tmp = Bcount << 16;
    int  Acount_tmp = Acount << 16;
    int  ABcount = Bcount_tmp | Acount;
    int  Excursion =  Acount_tmp | Acount;
    int  Numb_addr = Numb *0x4 ;
    int  OPT_aar   = Numb *0x20 ;
    int  Numb_TCC  = Numb <<12 ;
    int  OPT_worth = 0x00900004  |  Numb_TCC ;
    set_OPT_addr = OPT_offset + OPT_aar ;

    DMA_BASE = DMA0_BASE + DMA_Numb * 0x40000;

    *(int*)(DMA_BASE+DCHMAP_offset + Numb_addr)= OPT_aar;
    /* 队列配置开始 */
    DNUM_Temp = *(int*)(DMA_BASE+DMAQNUM_offset + (Numb/8)*0x4 );
    Numb_temp = Numb % 8 ;
    switch(Numb_temp)
    {
    case 0: DNUM_Temp = (DNUM_Temp & 0xFFFFFFF0) | QueueNUM; break;
    case 1: DNUM_Temp = (DNUM_Temp & 0xFFFFFF0F) | (QueueNUM << 4 ) ; break;
    case 2: DNUM_Temp = (DNUM_Temp & 0xFFFFF0FF) | (QueueNUM << 8 ) ; break;
    case 3: DNUM_Temp = (DNUM_Temp & 0xFFFF0FFF) | (QueueNUM << 12) ; break;
    case 4: DNUM_Temp = (DNUM_Temp & 0xFFF0FFFF) | (QueueNUM << 16) ; break;
    case 5: DNUM_Temp = (DNUM_Temp & 0xFF0FFFFF) | (QueueNUM << 20) ; break;
    case 6: DNUM_Temp = (DNUM_Temp & 0xF0FFFFFF) | (QueueNUM << 24) ; break;
    case 7: DNUM_Temp = (DNUM_Temp & 0x0FFFFFFF) | (QueueNUM << 28) ; break;
    default: break;
    }
    *(int*)(DMA_BASE+DMAQNUM_offset + (Numb/8)*0x4 ) = DNUM_Temp ;
    /*  队列配置结束     */

    /*  参数组配置     */
    *(int*)(DMA_BASE+set_OPT_addr)         = OPT_worth;
    *(int*)(DMA_BASE+set_OPT_addr+0x4)     = SourceAddr;
    *(int*)(DMA_BASE+set_OPT_addr+0x8)     = ABcount;
    *(int*)(DMA_BASE+set_OPT_addr+0xc)     = DestinationAddr;
    *(int*)(DMA_BASE+set_OPT_addr+0x10)    = Excursion;
    *(int*)(DMA_BASE+set_OPT_addr+0x14)    = 0X0001FFFF;
    *(int*)(DMA_BASE+set_OPT_addr+0x18)    = 0X00010001;
    *(int*)(DMA_BASE+set_OPT_addr+0x1c)    = 0X00010001;

    if(Numb <32)
        *(int*)(DMA_BASE+IESR_offset)      =  IntEnable;
    else
        *(int*)(DMA_BASE+IESR_offset+0x4)  =  IntEnable;
}


/**
********************************************************************************
 *   @n@b DMA_Start(volatile unsigned int DMA_Numb ,volatile unsigned int Numb)
 *
 *   @b 功能：
 *   @n DMA启动传输函数
 *
 *   @b 形参：
 *   @verbatim
 *       DMA_Numb 为DMA部件的选择，为DMA0和DMA2其中的一个
 *       Numb为通道号
 *   @endverbatim
 *
 *   <b> 返回值： </b>
 *   @n 无
 *
 *   <b> 注意事项： </b>
 *   @n 无
 *******************************************************************************
 */
void DMA_Start(volatile unsigned int DMA_Numb , volatile unsigned int Numb)
{
    volatile unsigned int DMA_BASE=0;
    DMA_BASE = DMA0_BASE + DMA_Numb * 0x40000;
    if(Numb <32)
    {
        int ESR_worth = 1 << Numb;
        *(int*)(DMA_BASE+EMCR_offset)     = ESR_worth;
        *(int*)(DMA_BASE+SECR_offset)     = ESR_worth;
        *(int*)(DMA_BASE+ESR_offset )     = ESR_worth;
    }
    else
    {
        Numb = Numb - 32 ;
        int  ESR_worth = 1 << Numb;
        *(int*)(DMA_BASE+EMCR_offset+0x4) = ESR_worth;
        *(int*)(DMA_BASE+SECR_offset+0x4) = ESR_worth;
        *(int*)(DMA_BASE+ESR_offset+0x4 ) = ESR_worth;
    }
}

/**
********************************************************************************
 *   @n@b DMA_TransState(volatile unsigned int DMA_Numb ,volatile unsigned int Numb)
 *
 *   @b 功能：
 *   @n DMA传输状态查询函数
 *
 *   @b 形参：
 *   @verbatim
 *       DMA_Numb 为DMA部件的选择，为DMA0和DMA2其中的一个
 *       Numb为通道号
 *   @endverbatim
 *
 *   <b> 返回值： </b>
 *   @n 若传输完成，返回数值1
 *
 *   <b> 注意事项： </b>
 *   @n 无
 *******************************************************************************
 */
int DMA_TransState(volatile unsigned int DMA_Numb , volatile unsigned int Numb)
{
    volatile int set_IPR_tmp = 0;
    volatile unsigned int DMA_BASE=0;
    DMA_BASE = DMA0_BASE + DMA_Numb * 0x40000;

    if(Numb <32)
    {
        int IPR_worth = 1 << Numb;
        while( set_IPR_tmp != IPR_worth )
        {
            set_IPR_tmp = *(int*)(DMA_BASE+IPR_offset);
            set_IPR_tmp = IPR_worth & set_IPR_tmp;
        }
        *(int*)(DMA_BASE+ICR_offset)  = IPR_worth;
    }
    else
    {
        Numb = Numb - 32 ;
        int IPR_worth = 1 << Numb;
        while( set_IPR_tmp != IPR_worth )
        {
            set_IPR_tmp = *(int*)(DMA_BASE+IPR_offset+0X4);
            set_IPR_tmp = IPR_worth & set_IPR_tmp;
        }
        *(int*)(DMA_BASE+ICR_offset+0X4) = IPR_worth;
    }

    return  1 ;
}
void dma_trans(void* src, void* dst,int size) {
   PSC_Open_Clk("DMA0" , 1);
   PSC_Open_Clk("DMA1" , 1);
   PSC_Open_Clk("DMA2" , 1);
   PSC_Open_Clk("DMA3" , 1);
   PSC_Open_Clk("DMA4" , 1);

   Uint32 temp_src = 0;
   Uint32 temp_dst = 0;
   int core_id = DNUM;
   temp_src = (Uint32)src;
   temp_dst = (Uint32)dst;
   if (size <= 0xffff) {
       DMA_ParaConfig(0, core_id ,0 ,temp_src ,size, 1, temp_dst ,0xFFFFFFFF);
       DMA_Start(0,core_id);
       while (!DMA_TransState(0,core_id));
   }
   else {
       int frame = size / 0x7fff;
      if (frame != 0) {
          temp_src = (Uint32)src;
          temp_dst = (Uint32)dst;
          DMA_ParaConfig(0, core_id ,0 ,temp_src ,0x7fff, frame, temp_dst ,0xFFFFFFFF);
          DMA_Start(0,core_id);
          while (!DMA_TransState(0,core_id));
      }
      if (size % 0x7fff != 0) {
         int size0 = size % 0x7fff;
         temp_src = (Uint32)src + 0x7fff * frame;
         temp_dst = (Uint32)dst + 0x7fff * frame;
        DMA_ParaConfig(0, core_id, 0, temp_src, size0, 1, temp_dst ,0xFFFFFFFF);
        DMA_Start(0,core_id);
        while (!DMA_TransState(0,core_id));
      }
   }
}
```
# softmax算子6678实现代码

```c
#include "softmax.h"
#include "78NE/utils.h"
#include "math.h"
// output = exp(input) / reduce_sum(exp(input), axis)
void fp_softmax_p(float *input_ptr, float *output_ptr, float *sum_data, int* param) {
    int axis = param[0];
    int n_dim = param[1];
    int inner_size = param[2];
    int outter_size = param[3];
    int axis_size = param[4];

  int i, j, k;
  for (i = 0; i < outter_size; i++) {
    int outter_offset = i * axis_size * inner_size;// input_shape[axis]: how many elements in one group
    int sum_outter_offset = i * inner_size;
    for (k = 0; k < inner_size; k++) {
      int inner_offset = outter_offset + k;
      float max_data = input_ptr[inner_offset];
      sum_data[k + sum_outter_offset] = 0;// for size of input_shape[axis]
      for (j = 0; j < axis_size; j++) {
        int axis_offset = inner_offset + j * inner_size;
        max_data = max_data > input_ptr[axis_offset] ? max_data : input_ptr[axis_offset];
      }
      for (j = 0; j < axis_size; j++) {
        int axis_offset = inner_offset + j * inner_size;
        output_ptr[axis_offset] = expf(input_ptr[axis_offset] - max_data);
        sum_data[k + sum_outter_offset] += output_ptr[axis_offset];
      }
    }
  }
  for (i = 0; i < outter_size; i++) {
    int outter_offset = i * axis_size * inner_size;
    int sum_outter_offset = i * inner_size;
    for (j = 0; j < axis_size; j++) {
      int axis_offset = outter_offset + j * inner_size;
      for (k = 0; k < inner_size; k++) {
        int inner_offset = axis_offset + k;
        output_ptr[inner_offset] = output_ptr[inner_offset] / sum_data[k + sum_outter_offset];
      }
    }
  }
}

float fp_max_p(float *input_ptr, int data_size, float existing_max, int step, int start)
{
    float max = existing_max;
    int i = start;
    for(i = start; i < data_size; i += step)
    {
        if(input_ptr[i] > max)
        {
            max = input_ptr[i];
        }
    }
    return max;
}

void fp_expf_p(float *input_ptr, float *output_ptr, float *sum_data,
                int data_size, float max_data, int step, int start)
{
    int i = start;// inner group id (in)
    sum_data[start] = 0;
    for(i = start; i < data_size; i += step)
    {
        output_ptr[i] = expf(input_ptr[i] - max_data);
        sum_data[start] += output_ptr[i];

    }
}

void fp_div_p(float *output_ptr, float sum, int data_size, int step, int start)
{
    int i = start;// inner group id (in)
    for(i = start; i < data_size; i += step)
    {
        output_ptr[i] = output_ptr[i] / sum;
    }
}


void i8_softmax_p(int8_t *input_ptr, float *output_ptr, float *sum_data, int* param) {
  int axis = param[0];
  int n_dim = param[1];
  int inner_size = param[2];
  int outter_size = param[3];
  int axis_size = param[4];
  int i, j, k;
  for (i = 0; i < outter_size; i++) {
    int outter_offset = i * axis_size * inner_size;// input_shape[axis]: how many elements in one group
    int sum_outter_offset = i * inner_size;
    for (k = 0; k < inner_size; k++) {
      int inner_offset = outter_offset + k;
      int8_t max_data = input_ptr[inner_offset];
      sum_data[k + sum_outter_offset] = 0;// for size of input_shape[axis]
      for (j = 0; j < axis_size; j++) {
        int axis_offset = inner_offset + j * inner_size;
        max_data = max_data > input_ptr[axis_offset] ? max_data : input_ptr[axis_offset];
      }
      for (j = 0; j < axis_size; j++) {
        int axis_offset = inner_offset + j * inner_size;
        output_ptr[axis_offset] = expf(input_ptr[axis_offset] - max_data);
        sum_data[k + sum_outter_offset] += output_ptr[axis_offset];
      }
    }
  }
  for (i = 0; i < outter_size; i++) {
    int outter_offset = i * axis_size * inner_size;
    int sum_outter_offset = i * inner_size;
    for (j = 0; j < axis_size; j++) {
      int axis_offset = outter_offset + j * inner_size;
      for (k = 0; k < inner_size; k++) {
        int inner_offset = axis_offset + k;
        output_ptr[inner_offset] = output_ptr[inner_offset] / sum_data[k + sum_outter_offset];
      }
    }
  }
}

float i8_max_p(int8_t *input_ptr, int data_size, float existing_max, int step, int start)
{
    float max = existing_max;
    int i = start;
    for(i = start; i < data_size; i += step)
    {
        if((float)(input_ptr[i]) > max)
        {
            max = (float)(input_ptr[i]);
        }
    }
    return max;
}

void i8_expf_p(int8_t *input_ptr, float *output_ptr, float *sum_data,
                int data_size, float max_data, int step, int start)
{
    int i = start;// inner group id (in)
    sum_data[start] = 0;
    for(i = start; i < data_size; i += step)
    {
        output_ptr[i] = expf((float)(input_ptr[i]) - max_data);
        sum_data[start] += output_ptr[i];

    }
}

void i8_div_p(float *output_ptr, float sum, int data_size, int step, int start)
{
    int i = start;// inner group id (in)
    for(i = start; i < data_size; i += step)
    {
        output_ptr[i] = output_ptr[i] / sum;
    }
}

// #define BLOCK_CAPACITY 76800 // floats
//int BLOCK_CAPACITY = 8; // floats

// divide the outter_size for different threads
void fp_softmax_s(float *input, float *output, float* sum_data,
                  int* param, int core_mask)
{
    int axis = param[0];
    int n_dim = param[1];
    int inner_size = param[2];
    int outter_size = param[3];
    int axis_size = param[4];

    // 锟斤拷锟斤拷锟斤拷锟睫改★拷锟斤拷通锟斤拷锟斤拷锟斤拷直锟斤拷锟狡碉拷每锟斤拷 outer 循锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟皆拷锟斤拷锟� (length / outter_size)
    int length_per_outer = inner_size * axis_size;

    int BLOCK_CAPACITY = 115200; // floats
    int core_id = DNUM;
    int logic_core_id = GetLogicCoreId(core_mask, core_id);

    int core_num = 0;
    core_num = GetCoreNum(core_mask);

    // 0x70800 Bytes --> 115200 floats
    float *inputtemp = (float *)(L2GADDBASE + logic_core_id*L2GADDOFFSET);
    float *outputtemp = (float *)(L2GADDBASE + logic_core_id*L2GADDOFFSET + 0x70800);
    float *sumdatatemp = (float *)(L2GADDBASE + logic_core_id*L2GADDOFFSET + 0xe1000);

    int outer_st = outter_size / core_num * logic_core_id;// assign outer loop to diff threads -> start id of outer loop
    unsigned int offset = outer_st * length_per_outer;// start id of each core, float
    int block_num = length_per_outer / BLOCK_CAPACITY; // block num of each outer loop

    int outer_num = (logic_core_id == core_num - 1) ? (outter_size - outer_st) : (outter_size / core_num);

    int curr_block;
    float* A_fixed = input;
    float* B_fixed = output;
    float* C_fixed = sum_data;

    int o = 0, in = 0;
    for(o = 0; o < outer_num; o ++)
    {
        int outer_id = outer_st + o;
        int outer_offset = outer_id * length_per_outer;
        int sum_outer_offset = outer_id * inner_size;
        for(in = 0; in < inner_size; in ++)
        {
            int inner_offset = outer_offset + in;

            // Step 1: get max value of this outer-inner group of axis_size data
            float max_data = input[inner_offset]; // get init value
            sum_data[sum_outer_offset + in] = 0;

            for(curr_block = 0; curr_block < block_num; curr_block ++)
            {
                A_fixed = &input[outer_offset + curr_block * BLOCK_CAPACITY];
                B_fixed = &output[outer_offset + curr_block * BLOCK_CAPACITY];
                // transfer data of current block: ddr --> L2 (input)
                dma_trans(A_fixed, inputtemp, BLOCK_CAPACITY * sizeof(float));
                float max_temp = fp_max_p(inputtemp, BLOCK_CAPACITY, max_data, inner_size, in);
                max_data = max_temp > max_data ? max_temp : max_data;
            }

            // process remained data (less than one BLOCK_CAPACITY)
            A_fixed = &input[outer_offset + curr_block * BLOCK_CAPACITY];
            B_fixed = &output[outer_offset + curr_block * BLOCK_CAPACITY];
            int remain_length = length_per_outer - BLOCK_CAPACITY * curr_block;
            if (remain_length > 0) {
                dma_trans(A_fixed, inputtemp, remain_length * sizeof(float));
                float max_temp = fp_max_p(inputtemp, remain_length, max_data, inner_size, in);
                max_data = max_temp > max_data ? max_temp : max_data;
            }

            // Step 2: use the value of max_data, do expf and calcu sum of one outer-inner group
            for(curr_block = 0; curr_block < block_num; curr_block ++)
            {
                A_fixed = &input[outer_offset + curr_block * BLOCK_CAPACITY];
                B_fixed = &output[outer_offset + curr_block * BLOCK_CAPACITY];
                // transfer data of current block: ddr --> L2 (input)
                dma_trans(A_fixed, inputtemp, BLOCK_CAPACITY * sizeof(float));
                fp_expf_p(inputtemp, outputtemp, sumdatatemp,
                            BLOCK_CAPACITY, max_data, inner_size, in); // calcu data of current block
                // transfer data of current block: L2 --> ddr (output)
                dma_trans(outputtemp, B_fixed, BLOCK_CAPACITY * sizeof(float));
                float sum_temp = sumdatatemp[in];
                sum_data[sum_outer_offset + in] += sum_temp;
            }

            // process remained data
            A_fixed = &input[outer_offset + curr_block * BLOCK_CAPACITY];
            B_fixed = &output[outer_offset + curr_block * BLOCK_CAPACITY];
            C_fixed = &sum_data[sum_outer_offset];
            remain_length = length_per_outer - BLOCK_CAPACITY * curr_block;

            if (remain_length > 0) {
                dma_trans(A_fixed, inputtemp, remain_length * sizeof(float));
                fp_expf_p(inputtemp, outputtemp, sumdatatemp,
                        remain_length, max_data, inner_size, in);
                dma_trans(outputtemp, B_fixed, remain_length * sizeof(float));
                float sum_temp = sumdatatemp[in];
                sum_data[sum_outer_offset + in] += sum_temp;
            }
        }
    }

    // Step 3: use the value of sum_data, do div
    for(o = 0; o < outer_num; o ++)
    {
        int outer_id = outer_st + o;
        int outer_offset = outer_id * length_per_outer;
        int sum_outer_offset = outer_id * inner_size;
        for(in = 0; in < inner_size; in ++)
        {
            for(curr_block = 0; curr_block < block_num; curr_block ++)
            {
                A_fixed = &input[outer_offset + curr_block * BLOCK_CAPACITY];
                B_fixed = &output[outer_offset + curr_block * BLOCK_CAPACITY];
                // transfer data of current block: ddr --> L2 (output)
                dma_trans(B_fixed, outputtemp, BLOCK_CAPACITY * sizeof(float));
                fp_div_p(outputtemp, sum_data[sum_outer_offset + in], BLOCK_CAPACITY, inner_size, in);
                // transfer data of current block: L2 --> ddr (output)
                dma_trans(outputtemp, B_fixed, BLOCK_CAPACITY * sizeof(float));
            }

            // process remained data
            A_fixed = &input[outer_offset + curr_block * BLOCK_CAPACITY];
            B_fixed = &output[outer_offset + curr_block * BLOCK_CAPACITY];
            int remain_length = length_per_outer - BLOCK_CAPACITY * curr_block;

            if (remain_length > 0) {
                dma_trans(B_fixed, outputtemp, remain_length * sizeof(float));
                fp_div_p(outputtemp, sum_data[sum_outer_offset + in], remain_length, inner_size, in);
                dma_trans(outputtemp, B_fixed, remain_length * sizeof(float));
            }
        }
    }
}

void i8_softmax_s(int8_t *input, float *output, float* sum_data,
                  int* param, int core_mask)
{
    int axis = param[0];
    int n_dim = param[1];
    int inner_size = param[2];
    int outter_size = param[3];
    int axis_size = param[4];

    int length_per_outer = inner_size * axis_size;

    // 【修复1】：将 Block 容量限制在安全范围内，适配 float 的 4 字节膨胀
    // 32768 元素对应 L2 占用：Input(32KB) + Output(131KB) + Sum(4KB) = 167KB 绝对安全
    int BLOCK_CAPACITY = 32768;

    int core_id = DNUM;
    int logic_core_id = GetLogicCoreId(core_mask, core_id);
    int core_num = GetCoreNum(core_mask);

    // 【修复2】：杜绝硬编码偏移，使用 sizeof() 动态推导连续的 L2 内存，绝不溢出
    int8_t *inputtemp = (int8_t *)(L2GADDBASE + logic_core_id*L2GADDOFFSET);
    float *outputtemp = (float *)((char*)inputtemp + BLOCK_CAPACITY * sizeof(int8_t));
    float *sumdatatemp = (float *)((char*)outputtemp + BLOCK_CAPACITY * sizeof(float));

    int outer_st = outter_size / core_num * logic_core_id;
    int block_num = length_per_outer / BLOCK_CAPACITY;

    // 尾核任务保护，防止 outter_size 不能被 core_num 整除
    int outer_num = (logic_core_id == core_num - 1) ? (outter_size - outer_st) : (outter_size / core_num);

    int curr_block;
    int o = 0, in = 0;

    for(o = 0; o < outer_num; o ++)
    {
        int outer_id = outer_st + o;
        int outer_offset = outer_id * length_per_outer;
        int sum_outer_offset = outer_id * inner_size;

        for(in = 0; in < inner_size; in ++)
        {
            int inner_offset = outer_offset + in;

            // Step 1: get max value
            float max_data = (float)input[inner_offset];
            sum_data[sum_outer_offset + in] = 0;

            for(curr_block = 0; curr_block < block_num; curr_block ++)
            {
                int8_t* A_fixed = &input[outer_offset + curr_block * BLOCK_CAPACITY];
                dma_trans(A_fixed, inputtemp, BLOCK_CAPACITY * sizeof(int8_t));
                float max_temp = i8_max_p(inputtemp, BLOCK_CAPACITY, max_data, inner_size, in);
                max_data = max_temp > max_data ? max_temp : max_data;
            }

            int remain_length = length_per_outer - BLOCK_CAPACITY * block_num;
            if (remain_length > 0) {
                int8_t* A_fixed = &input[outer_offset + block_num * BLOCK_CAPACITY];
                dma_trans(A_fixed, inputtemp, remain_length * sizeof(int8_t));
                float max_temp = i8_max_p(inputtemp, remain_length, max_data, inner_size, in);
                max_data = max_temp > max_data ? max_temp : max_data;
            }

            // Step 2: expf and calcu sum
            for(curr_block = 0; curr_block < block_num; curr_block ++)
            {
                int8_t* A_fixed = &input[outer_offset + curr_block * BLOCK_CAPACITY];
                float* B_fixed = &output[outer_offset + curr_block * BLOCK_CAPACITY];

                dma_trans(A_fixed, inputtemp, BLOCK_CAPACITY * sizeof(int8_t));

                // 【修复3】：在计算当前 in 之前，强制将输出从 DDR 取回 L2！
                // 这一步完美防止了 L2 的未初始化区域在后续 DMA 写回时覆盖掉 DDR 中其他 in 的正确值
                dma_trans(B_fixed, outputtemp, BLOCK_CAPACITY * sizeof(float));
                // 【核心修复】：在每次内核调用前，强制清零 L2 临时累加器！
                sumdatatemp[in] = 0.0f;
                i8_expf_p(inputtemp, outputtemp, sumdatatemp, BLOCK_CAPACITY, max_data, inner_size, in);

                dma_trans(outputtemp, B_fixed, BLOCK_CAPACITY * sizeof(float));

                float sum_temp = sumdatatemp[in];
                sum_data[sum_outer_offset + in] += sum_temp;
            }

            if (remain_length > 0) {
                int8_t* A_fixed = &input[outer_offset + block_num * BLOCK_CAPACITY];
                float* B_fixed = &output[outer_offset + block_num * BLOCK_CAPACITY];

                dma_trans(A_fixed, inputtemp, remain_length * sizeof(int8_t));

                // 【修复3】：同样补充剩余长度的 DDR 取回
                dma_trans(B_fixed, outputtemp, remain_length * sizeof(float));
                // 【核心修复】：在每次内核调用前，强制清零 L2 临时累加器！
                sumdatatemp[in] = 0.0f;
                i8_expf_p(inputtemp, outputtemp, sumdatatemp, remain_length, max_data, inner_size, in);

                dma_trans(outputtemp, B_fixed, remain_length * sizeof(float));

                float sum_temp = sumdatatemp[in];
                sum_data[sum_outer_offset + in] += sum_temp;
            }
        }
    }

    // Step 3: div
    for(o = 0; o < outer_num; o ++)
    {
        int outer_id = outer_st + o;
        int outer_offset = outer_id * length_per_outer;
        int sum_outer_offset = outer_id * inner_size;

        for(in = 0; in < inner_size; in ++)
        {
            for(curr_block = 0; curr_block < block_num; curr_block ++)
            {
                float* B_fixed = &output[outer_offset + curr_block * BLOCK_CAPACITY];
                dma_trans(B_fixed, outputtemp, BLOCK_CAPACITY * sizeof(float));
                i8_div_p(outputtemp, sum_data[sum_outer_offset + in], BLOCK_CAPACITY, inner_size, in);
                dma_trans(outputtemp, B_fixed, BLOCK_CAPACITY * sizeof(float));
            }

            int remain_length = length_per_outer - BLOCK_CAPACITY * block_num;
            if (remain_length > 0) {
                float* B_fixed = &output[outer_offset + block_num * BLOCK_CAPACITY];
                dma_trans(B_fixed, outputtemp, remain_length * sizeof(float));
                i8_div_p(outputtemp, sum_data[sum_outer_offset + in], remain_length, inner_size, in);
                dma_trans(outputtemp, B_fixed, remain_length * sizeof(float));
            }
        }
    }
}

```

## LSTM 6678实现（含单核、多核）
```
void UpdateState_s(float *cell_state, float *forget_gate, float *input_gate, float *cell_gate,
                 float *state_buffer, int batch, int hidden_size, float zoneout, int core_mask)
{
    int num = GetCoreNum(core_mask);
    int logic_core_id = GetLogicCoreId(core_mask, DNUM);
    if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON))
    {
        if(logic_core_id == 0)
            dma_trans( (void*) cell_state, (void*) state_buffer,batch * hidden_size * sizeof(float));
        C6678E_SyncN(num, logic_core_id);
        ElementOptMulSMCFp32(state_buffer, &zoneout, state_buffer, batch * hidden_size, 0, core_mask);
    }
    C6678E_SyncN(num, logic_core_id);
    ElementMulSMCFp32(forget_gate, cell_state, cell_state, batch * hidden_size, core_mask);
    C6678E_SyncN(num, logic_core_id);
    ElementMulAccSMCFp32(input_gate, cell_gate, cell_state, batch * hidden_size, core_mask);
    C6678E_SyncN(num, logic_core_id);
    if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON))
    {
        ElementOptMulAccSMCFp32(cell_state, 1 - zoneout, state_buffer, batch * hidden_size, core_mask);
    }
    C6678E_SyncN(num, logic_core_id);
}
void LstmMatMul(float *c, float *a, float *b, float *bias, int row, int deep, int col, int col_align,
                int is_vec, float *packed_ptr)
{
    if(is_vec)
    {
        MatVecMulFp32(a, b, c, bias, ActType_No, deep, col);
    }
    else
    {
        MatMul12x8(a, b, c, bias, ActType_No, deep, row, col, col, OutType_Nhwc);
    }
}
void PackLstmInput(const float *src, float *dst, int row, int deep)
{
    RowMajor2Col12Major(src, dst, row, deep, 0, row);
}

void UpdateOutput_s(float *hidden_state, float *output, const float *cell_state, const float *output_gate,
                  float *weight_project, float *buffer[8], LstmParameter *lstm_param, int core_mask)
{
    int batch = lstm_param->batch_;
    int hidden_size = lstm_param->hidden_size_;
    int output_size = lstm_param->output_size_;
    float *state_buffer = buffer[4];
    float *hidden_buffer = weight_project ? buffer[2] : hidden_state;
    float zoneout = lstm_param->zoneout_hidden_;
    int core_id = DNUM;
    int logic_core_id = GetLogicCoreId(core_mask, core_id);
    int num = GetCoreNum(core_mask);
    C6678E_SyncN(num, logic_core_id);
    if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON))
    {
        if(logic_core_id == 0)
            dma_trans( (void*)hidden_state, (void*)state_buffer,batch * output_size * sizeof(float));
        C6678E_SyncN(num, logic_core_id);
        ElementOptMulSMCFp32(state_buffer, &zoneout, state_buffer, batch * output_size, 0, core_mask);
    }
    C6678E_SyncN(num, logic_core_id);
    TanhSMCFp32(cell_state, hidden_buffer,  batch * hidden_size, core_mask);
    C6678E_SyncN(num, logic_core_id);
    ElementMulSMCFp32(hidden_buffer, output_gate, hidden_buffer, batch * hidden_size, core_mask);
    C6678E_SyncN(num, logic_core_id);

    if(logic_core_id == 0)
    {
        if (weight_project)
        {
            float *left_matrix = hidden_buffer;
            if (batch != 1)
            {
                left_matrix = buffer[6];
                PackLstmInput(hidden_buffer, left_matrix, batch, hidden_size);
            }
            LstmMatMul(hidden_state, left_matrix, weight_project, NULL, batch, hidden_size, output_size,
                       lstm_param->proj_col_align_, batch == 1, buffer[7]);
        }
    }
    C6678E_SyncN(num, logic_core_id);
    if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON))
    {
        ElementOptMulAccSMCFp32(hidden_state, 1 - zoneout, state_buffer, batch * output_size, core_mask);
    }
    C6678E_SyncN(num, logic_core_id);
    if(logic_core_id == 0)
        dma_trans( (void*)hidden_state, (void*)output,batch * output_size * sizeof(float));
    C6678E_SyncN(num, logic_core_id);
}


void UpdateLstmGate(float *gate_buffer, const float *input, const float *weight, const float *bias, int row, int deep,
                    int col, int col_align, int is_vec, float *packed_ptr)
{
    const float *weight_i = weight;
    const float *bias_i = bias;
    float *gate_i = gate_buffer;
    int i = 0;
    for (i = 0; i < 4; i++)
    {
        LstmMatMul(gate_i, input, weight_i, bias_i, row, deep, col, col_align, is_vec, packed_ptr);
        weight_i += deep * (is_vec ? col : col_align);
        bias_i += col_align;
        gate_i += row * col;
    }
}

void LstmStepUnit_s(float *output, float *input_gate, float *forget_gate, float *cell_gate, float *output_gate,
                  const float *state_weight, const float *state_bias, const float *weight_project, float *hidden_state,
                  float *cell_state, float *buffer[8],
                  const LstmParameter *lstm_param, int core_mask)
{
    float *packed_state = buffer[1];
    float *state_gate = buffer[2];
    float *cell_buffer = buffer[3];
    float *hidden_buffer = buffer[4];
    float *packed_output = buffer[5];
    bool is_vec = lstm_param->batch_ == 1;
    int core_id = DNUM;
    int logic_core_id = GetLogicCoreId(core_mask, core_id);
    int num = GetCoreNum(core_mask);
    C6678E_SyncN(num, logic_core_id);



    if(logic_core_id == 0)
    {
        if (is_vec)
        {
            UpdateLstmGate(state_gate, hidden_state, state_weight, state_bias, lstm_param->batch_, lstm_param->output_size_,
                               lstm_param->hidden_size_, lstm_param->state_col_align_, is_vec, packed_output);
        }
        else
        {

            PackLstmInput(hidden_state, packed_state, lstm_param->batch_, lstm_param->output_size_);
            UpdateLstmGate(state_gate, packed_state, state_weight, state_bias, lstm_param->batch_, lstm_param->output_size_,
                           lstm_param->hidden_size_, lstm_param->state_col_align_, is_vec, packed_output);
        }
    }
    C6678E_SyncN(num, logic_core_id);

    ElementAddFp32SMC(input_gate, state_gate, input_gate, lstm_param->batch_ * lstm_param->hidden_size_, core_mask);
    ElementAddFp32SMC(forget_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_ * 2, forget_gate,
             lstm_param->batch_ * lstm_param->hidden_size_, core_mask);
    ElementAddFp32SMC(cell_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_ * 3, cell_gate,
             lstm_param->batch_ * lstm_param->hidden_size_, core_mask);
    ElementAddFp32SMC(output_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_, output_gate,
             lstm_param->batch_ * lstm_param->hidden_size_, core_mask);
    C6678E_SyncN(num, logic_core_id);

    SigmoidSMCFp32(input_gate, input_gate, lstm_param->batch_ * lstm_param->hidden_size_, core_mask);
    SigmoidSMCFp32(forget_gate, forget_gate, lstm_param->batch_ * lstm_param->hidden_size_, core_mask);
    TanhSMCFp32(cell_gate, cell_gate, lstm_param->batch_ * lstm_param->hidden_size_, core_mask);
    C6678E_SyncN(num, logic_core_id);
    UpdateState_s(cell_state, forget_gate, input_gate, cell_gate, cell_buffer, lstm_param->batch_, lstm_param->hidden_size_,
              lstm_param->zoneout_cell_, core_mask);

    SigmoidSMCFp32(output_gate, output_gate, lstm_param->batch_ * lstm_param->hidden_size_, core_mask);
    C6678E_SyncN(num, logic_core_id);

    UpdateOutput_s(hidden_state, output, cell_state, output_gate, weight_project, buffer, lstm_param, core_mask);
    C6678E_SyncN(num, logic_core_id);
    if(logic_core_id == 0)
    {
        if (!(lstm_param->zoneout_cell_ >= -FLT_EPSILON && lstm_param->zoneout_cell_ <= FLT_EPSILON))
        {
            dma_trans( (void*)cell_buffer, (void*)cell_state,lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float));
        }

        if (!(lstm_param->zoneout_hidden_ >= -FLT_EPSILON && lstm_param->zoneout_hidden_ <= FLT_EPSILON))
        {
            dma_trans( (void*)hidden_buffer, (void*)hidden_state,lstm_param->batch_ * lstm_param->output_size_ * sizeof(float));
        }
    }
    C6678E_SyncN(num, logic_core_id);
}
void LstmUnidirectional_s(float *output, const float *packed_input, const float *weight_i, const float *weight_h,
                        const float *input_bias, const float *state_bias, float *hidden_state, float *cell_state,
                        float *buffer[8], const LstmParameter *lstm_param, int is_backward, int core_mask)
{
    float *gate = buffer[0];
    int i = 0;
    int core_id = DNUM;
    int logic_core_id = GetLogicCoreId(core_mask, core_id);
    int core_num = GetCoreNum(core_mask);

    if( logic_core_id == 0)
    {
        for ( i = 0; i < 4; i++)
        {
            const float *weight_loop = weight_i + lstm_param->input_size_ * lstm_param->input_col_align_ * i;
            const float *bias_loop = input_bias + lstm_param->input_col_align_ * i;
            float *gate_loop = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * i;
            MatMul12x8(packed_input, weight_loop, gate_loop, bias_loop, ActType_No, lstm_param->input_size_,
                      lstm_param->seq_len_ * lstm_param->batch_,
                       lstm_param->hidden_size_,
                        lstm_param->hidden_size_,
                      OutType_Nhwc);
        }
    }

    C6678E_SyncN(core_num, logic_core_id);

    float *input_gate = gate;
    float *forget_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * 2;
    float *cell_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * 3;
    float *output_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_;
    int t = 0;
    for (t = 0; t < lstm_param->seq_len_; t++)
    {
        int real_t = is_backward ? lstm_param->seq_len_ - t - 1 : t;
        float *input_gate_t = input_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
        float *forget_gate_t = forget_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
        float *cell_gate_t = cell_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
        float *output_gate_t = output_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
        float *output_ptr = output + real_t * lstm_param->output_step_;
        LstmStepUnit_s(output_ptr, input_gate_t, forget_gate_t, cell_gate_t, output_gate_t, weight_h, state_bias, NULL,
                     hidden_state, cell_state, buffer, lstm_param, core_mask);
    }
}


void fp_lstm_s(float *output, const float *input, unsigned long long *params, int core_mask)
{
    float *weight_i = (float *)params[0];
    float *weight_h = (float *)params[1];
    float *input_bias = (float *)params[2];
    float *state_bias = (float *)params[3];
    float *hidden_state = (float *)params[4];
    float *cell_state = (float *)params[5];
    float **buffer = (float **)params[6];
    LstmParameter *lstm_param = (LstmParameter *)params[7];

    float *packed_input = buffer[0];
    buffer += 1;
    int core_id = DNUM;
    int logic_core_id = GetLogicCoreId(core_mask, core_id);
    int core_num = GetCoreNum(core_mask);
    if( logic_core_id == 0)
    {
        PackLstmInput(input, packed_input, lstm_param->seq_len_ * lstm_param->batch_, lstm_param->input_size_);
    }
    C6678E_SyncN(core_num, logic_core_id);
    LstmUnidirectional_s(output, packed_input, weight_i, weight_h, input_bias, state_bias, hidden_state, cell_state, buffer,
                     lstm_param, false, core_mask);



    C6678E_SyncN(core_num, logic_core_id);

    if (lstm_param->bidirectional_)
    {
        const float *backward_weight_i = weight_i + 4 * lstm_param->input_col_align_ * lstm_param->input_size_;
        const float *backward_weight_h = weight_h + 4 * lstm_param->state_col_align_ * lstm_param->output_size_;
        const float *backward_input_bias = input_bias + 4 * lstm_param->input_col_align_;
        const float *backward_state_bias = state_bias + 4 * lstm_param->state_col_align_;
        float *backward_output = output + lstm_param->batch_ * lstm_param->output_size_;
        float *backward_cell_state = cell_state + lstm_param->batch_ * lstm_param->hidden_size_;
        float *backward_hidden_state = hidden_state + lstm_param->batch_ * lstm_param->output_size_;






        LstmUnidirectional_s(backward_output, packed_input, backward_weight_i, backward_weight_h, backward_input_bias,
                           backward_state_bias, backward_hidden_state, backward_cell_state, buffer, lstm_param, true, core_mask);
      }
    C6678E_SyncN(core_num, logic_core_id);
}

void UpdateState(float *cell_state, float *forget_gate, float *input_gate, float *cell_gate,
                 float *state_buffer, int batch, int hidden_size, float zoneout)
{
    if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON))
    {
        dma_trans( (void*) cell_state, (void*) state_buffer,batch * hidden_size * sizeof(float));
        ElementOptMulFp32(state_buffer, &zoneout, state_buffer, batch * hidden_size, 0);
    }
    ElementMulFp32(forget_gate, cell_state, cell_state, batch * hidden_size);
    ElementMulAcc(input_gate, cell_gate, cell_state, batch * hidden_size);
    if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON))
    {
        ElementOptMulAcc(cell_state, 1 - zoneout, state_buffer, batch * hidden_size);
    }
}
void UpdateOutput(float *hidden_state, float *output, const float *cell_state, const float *output_gate,
                  float *weight_project, float *buffer[8], LstmParameter *lstm_param)
{
    int batch = lstm_param->batch_;
    int hidden_size = lstm_param->hidden_size_;
    int output_size = lstm_param->output_size_;
    float *state_buffer = buffer[4];
    float *hidden_buffer = weight_project ? buffer[2] : hidden_state;
    float zoneout = lstm_param->zoneout_hidden_;
    if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON))
    {

        dma_trans( (void*)hidden_state, (void*)state_buffer,batch * output_size * sizeof(float));
        ElementOptMulFp32(state_buffer, &zoneout, state_buffer, batch * output_size, 0);
    }

    TanhFp32(cell_state, hidden_buffer,  batch * hidden_size);
    ElementMulFp32(hidden_buffer, output_gate, hidden_buffer, batch * hidden_size);
    if (weight_project)
    {
        float *left_matrix = hidden_buffer;
        if (batch != 1)
        {
            left_matrix = buffer[6];
            PackLstmInput(hidden_buffer, left_matrix, batch, hidden_size);
        }
        LstmMatMul(hidden_state, left_matrix, weight_project, NULL, batch, hidden_size, output_size,
                   lstm_param->proj_col_align_, batch == 1, buffer[7]);
    }
    if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON))
    {
        ElementOptMulAcc(hidden_state, 1 - zoneout, state_buffer, batch * output_size);
    }

    dma_trans( (void*)hidden_state, (void*)output,batch * output_size * sizeof(float));
}
void LstmStepUnit(float *output, float *input_gate, float *forget_gate, float *cell_gate, float *output_gate,
                  const float *state_weight, const float *state_bias, const float *weight_project, float *hidden_state,
                  float *cell_state, float *buffer[8],
                  const LstmParameter *lstm_param)
{
    float *packed_state = buffer[1];
    float *state_gate = buffer[2];
    float *cell_buffer = buffer[3];
    float *hidden_buffer = buffer[4];
    float *packed_output = buffer[5];
    bool is_vec = lstm_param->batch_ == 1;

    if (is_vec)
    {
        UpdateLstmGate(state_gate, hidden_state, state_weight, state_bias, lstm_param->batch_, lstm_param->output_size_,
                           lstm_param->hidden_size_, lstm_param->state_col_align_, is_vec, packed_output);
    }
    else
    {

        PackLstmInput(hidden_state, packed_state, lstm_param->batch_, lstm_param->output_size_);
        UpdateLstmGate(state_gate, packed_state, state_weight, state_bias, lstm_param->batch_, lstm_param->output_size_,
                       lstm_param->hidden_size_, lstm_param->state_col_align_, is_vec, packed_output);
    }

    ElementAddFp32(input_gate, state_gate, input_gate, lstm_param->batch_ * lstm_param->hidden_size_);
    ElementAddFp32(forget_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_ * 2, forget_gate,
             lstm_param->batch_ * lstm_param->hidden_size_);
    ElementAddFp32(cell_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_ * 3, cell_gate,
             lstm_param->batch_ * lstm_param->hidden_size_);
    ElementAddFp32(output_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_, output_gate,
             lstm_param->batch_ * lstm_param->hidden_size_);
    SigmoidFp32(input_gate, input_gate, lstm_param->batch_ * lstm_param->hidden_size_);
    SigmoidFp32(forget_gate, forget_gate, lstm_param->batch_ * lstm_param->hidden_size_);
    TanhFp32(cell_gate, cell_gate, lstm_param->batch_ * lstm_param->hidden_size_);
    UpdateState(cell_state, forget_gate, input_gate, cell_gate, cell_buffer, lstm_param->batch_, lstm_param->hidden_size_,
              lstm_param->zoneout_cell_);
    SigmoidFp32(output_gate, output_gate, lstm_param->batch_ * lstm_param->hidden_size_);
    UpdateOutput(hidden_state, output, cell_state, output_gate, weight_project, buffer, lstm_param);
    if (!(lstm_param->zoneout_cell_ >= -FLT_EPSILON && lstm_param->zoneout_cell_ <= FLT_EPSILON))
    {
        dma_trans( (void*)cell_buffer, (void*)cell_state,lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float));
    }

    if (!(lstm_param->zoneout_hidden_ >= -FLT_EPSILON && lstm_param->zoneout_hidden_ <= FLT_EPSILON))
    {
        dma_trans( (void*)hidden_buffer, (void*)hidden_state,lstm_param->batch_ * lstm_param->output_size_ * sizeof(float));
    }
}
void LstmUnidirectional(float *output, const float *packed_input, const float *weight_i, const float *weight_h,
                        const float *input_bias, const float *state_bias, float *hidden_state, float *cell_state,
                        float *buffer[8], const LstmParameter *lstm_param, int is_backward)
{

    float *gate = buffer[0];
    int i = 0;

    for ( i = 0; i < 4; i++)
    {
        const float *weight_loop = weight_i + lstm_param->input_size_ * lstm_param->input_col_align_ * i;
        const float *bias_loop = input_bias + lstm_param->input_col_align_ * i;
        float *gate_loop = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * i;
        MatMul12x8(packed_input, weight_loop, gate_loop, bias_loop, ActType_No, lstm_param->input_size_,
                  lstm_param->seq_len_ * lstm_param->batch_,
                   lstm_param->hidden_size_,
                    lstm_param->hidden_size_,
                  OutType_Nhwc);
    }

    float *input_gate = gate;
    float *forget_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * 2;
    float *cell_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * 3;
    float *output_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_;
    int t = 0;
    for (t = 0; t < lstm_param->seq_len_; t++)
    {
        int real_t = is_backward ? lstm_param->seq_len_ - t - 1 : t;
        float *input_gate_t = input_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
        float *forget_gate_t = forget_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
        float *cell_gate_t = cell_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
        float *output_gate_t = output_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
        float *output_ptr = output + real_t * lstm_param->output_step_;
        LstmStepUnit(output_ptr, input_gate_t, forget_gate_t, cell_gate_t, output_gate_t, weight_h, state_bias, NULL,
                     hidden_state, cell_state, buffer, lstm_param);
    }
}
void fp_lstm_p(float *output, const float *input, unsigned long long * params)
{
    float *weight_i = (float *)params[0];
    float *weight_h = (float *)params[1];
    float *input_bias = (float *)params[2];
    float *state_bias = (float *)params[3];
    float *hidden_state = (float *)params[4];
    float *cell_state = (float *)params[5];
    float **buffer = (float *)params[6];
    LstmParameter *lstm_param = (LstmParameter *)params[7];

    float *packed_input = buffer[0];
    buffer += 1;
    PackLstmInput(input, packed_input, lstm_param->seq_len_ * lstm_param->batch_, lstm_param->input_size_);
    LstmUnidirectional(output, packed_input, weight_i, weight_h, input_bias, state_bias, hidden_state, cell_state, buffer,
                     lstm_param, false);
    if (lstm_param->bidirectional_)
    {
        const float *backward_weight_i = weight_i + 4 * lstm_param->input_col_align_ * lstm_param->input_size_;
        const float *backward_weight_h = weight_h + 4 * lstm_param->state_col_align_ * lstm_param->output_size_;
        const float *backward_input_bias = input_bias + 4 * lstm_param->input_col_align_;
        const float *backward_state_bias = state_bias + 4 * lstm_param->state_col_align_;
        float *backward_output = output + lstm_param->batch_ * lstm_param->output_size_;
        float *backward_cell_state = cell_state + lstm_param->batch_ * lstm_param->hidden_size_;
        float *backward_hidden_state = hidden_state + lstm_param->batch_ * lstm_param->output_size_;
        LstmUnidirectional(backward_output, packed_input, backward_weight_i, backward_weight_h, backward_input_bias,
                           backward_state_bias, backward_hidden_state, backward_cell_state, buffer, lstm_param, true);
      }
}

```

## ElementEqual和ElementGreaterFloor算子：

```
int get_total_elements(int num_dims, int *dims) {
    if (num_dims == 0) return 1;
    int total = 1;
    int i;
    for (i = 0; i < num_dims; ++i) {
        total *= dims[i];
    }
    return total;
}

// --- 1. 单核私有空间算子 (Private) ---
void fp_greater_equal_p(float *input0, float *input1, bool *output, unsigned long long *param) {
    int* input0_dims = (int*) (uint32_t) param[0];
    int* input1_dims = (int*) (uint32_t) param[1];
    int* output_dims = (int*) (uint32_t) param[2];
    int* strides0 = (int*) (uint32_t) param[3];
    int* strides1 = (int*) (uint32_t) param[4];
    int* strides_output = (int*) (uint32_t) param[5];
    int num_dims = (int) param[6];
    
    int same_shape = 1;
    int i;
    for (i = 0; i < num_dims; ++i) {
        if (input0_dims[i] != input1_dims[i]) {
            same_shape = 0;
            break;
        }
    }
    
    int input0_elements_num = get_total_elements(num_dims, input0_dims);
    int input1_elements_num = get_total_elements(num_dims, input1_dims);

    // Case 1: 维度相同，逐元素比较
    if (same_shape) {
        for (i = 0; i < input0_elements_num; i++) {
            output[i] = input0[i] >= input1[i];
        }
    } 
    // Case 2: input0 是标量
    else if (input0_elements_num == 1) {
        float in0_val = input0[0];
        for (i = 0; i < input1_elements_num; i++) {
            output[i] = in0_val >= input1[i];
        }
    } 
    // Case 3: input1 是标量
    else if (input1_elements_num == 1) {
        float in1_val = input1[0];
        for (i = 0; i < input0_elements_num; i++) {
            output[i] = input0[i] >= in1_val;
        }
    } 
    // Case 4: 广播模式
    else {
        int total_elements = get_total_elements(num_dims, output_dims);
        
        strides0[num_dims - 1] = 1;
        strides1[num_dims - 1] = 1;
        strides_output[num_dims - 1] = 1;
        for (i = num_dims - 2; i >= 0; --i) {
            strides0[i] = strides0[i + 1] * input0_dims[i + 1];
            strides1[i] = strides1[i + 1] * input1_dims[i + 1];
            strides_output[i] = strides_output[i + 1] * output_dims[i + 1];
        }

        for (i = 0; i < total_elements; ++i) {
            int offset0 = 0;
            int offset1 = 0;
            int temp_i = i;
            int d;
            for (d = 0; d < num_dims; ++d) {
                int coord = temp_i / strides_output[d];
                temp_i %= strides_output[d];

                if (input0_dims[d] > 1) {
                    offset0 += coord * strides0[d];
                }
                if (input1_dims[d] > 1) {
                    offset1 += coord * strides1[d];
                }
            }
            output[i] = input0[offset0] >= input1[offset1];
        }
    }
}
// --- 2. 多核共享空间算子 (Shared) ---
void fp_greater_equal_s(float *input0, float *input1, bool *output, unsigned long long *param, int core_mask) {
    int core_id = get_core_id();
    int logic_core_id = GetLogicCoreId(core_mask, core_id);
    int core_num = GetCoreNum(core_mask);

    int* input0_dims = (int*) (uint32_t) param[0];
    int* input1_dims = (int*) (uint32_t) param[1];
    int* output_dims = (int*) (uint32_t) param[2];
    int* strides0 = (int*) (uint32_t) param[3];
    int* strides1 = (int*) (uint32_t) param[4];
    int* strides_output = (int*) (uint32_t) param[5];
    int num_dims = (int) param[6];
    
    int same_shape = 1;
    int i;
    for (i = 0; i < num_dims; ++i) {
        if (input0_dims[i] != input1_dims[i]) {
            same_shape = 0;
            break;
        }
    }
    
    int input0_elements_num = get_total_elements(num_dims, input0_dims);
    int input1_elements_num = get_total_elements(num_dims, input1_dims);

    // Case 1: 维度相同，2输入1输出，借用L2+DMA加速
    if (same_shape) {
        int size = input0_elements_num;
        int elem_per_core = (size + core_num - 1) / core_num;
        int start_index = logic_core_id * elem_per_core;
        int end_index = min(start_index + elem_per_core, size);
        int elem_this_core = end_index - start_index;

        if (elem_this_core <= 0) return;

        float* ddr_in0 = input0 + start_index;
        float* ddr_in1 = input1 + start_index;
        bool* ddr_out = output + start_index;

        // L2分3块（2块float，1块bool）
        int l2_block_size = l2_space_size / 3;
        // 以最大类型(float)保证安全容量
        int l2_block_capacity = (l2_block_size - l2_block_padding) / sizeof(float);

        float* l2_in0 = (float*)get_l2_addr(core_id);
        float* l2_in1 = (float*)((char*)get_l2_addr(core_id) + l2_block_size);
        bool* l2_out = (bool*)((char*)get_l2_addr(core_id) + l2_block_size * 2);

        int elem_left = elem_this_core;

        while (elem_left > 0) {
            int elem_per_iter = min(elem_left, l2_block_capacity);
            int in_dma_size = elem_per_iter * sizeof(float);
            int out_dma_size = elem_per_iter * sizeof(bool);

            dma_trans(ddr_in0, l2_in0, in_dma_size);
            dma_trans(ddr_in1, l2_in1, in_dma_size);

            for (i = 0; i < elem_per_iter; i++) {
                l2_out[i] = l2_in0[i] >= l2_in1[i];
            }

            dma_trans(l2_out, ddr_out, out_dma_size);

            ddr_in0 += elem_per_iter;
            ddr_in1 += elem_per_iter;
            ddr_out += elem_per_iter;
            elem_left -= elem_per_iter;
        }
    } 
    // Case 2: 某个端是标量（1输入1输出，本质），借用L2+DMA加速
    else if (input0_elements_num == 1 || input1_elements_num == 1) {
        int size = (input0_elements_num == 1) ? input1_elements_num : input0_elements_num;
        int in0_is_scalar = (input0_elements_num == 1);
        float scalar_val = in0_is_scalar ? input0[0] : input1[0];
        float* ddr_in = in0_is_scalar ? input1 : input0;

        int elem_per_core = (size + core_num - 1) / core_num;
        int start_index = logic_core_id * elem_per_core;
        int end_index = min(start_index + elem_per_core, size);
        int elem_this_core = end_index - start_index;

        if (elem_this_core <= 0) return;

        ddr_in += start_index;
        bool* ddr_out = output + start_index;

        int l2_block_size = l2_space_size / 2;
        int l2_block_capacity = (l2_block_size - l2_block_padding) / sizeof(float);

        float* l2_in = (float*)get_l2_addr(core_id);
        bool* l2_out = (bool*)((char*)get_l2_addr(core_id) + l2_block_size);

        int elem_left = elem_this_core;

        while (elem_left > 0) {
            int elem_per_iter = min(elem_left, l2_block_capacity);
            int in_dma_size = elem_per_iter * sizeof(float);
            int out_dma_size = elem_per_iter * sizeof(bool);

            dma_trans(ddr_in, l2_in, in_dma_size);

            for (i = 0; i < elem_per_iter; i++) {
                if (in0_is_scalar) {
                    l2_out[i] = scalar_val >= l2_in[i];
                } else {
                    l2_out[i] = l2_in[i] >= scalar_val;
                }
            }

            dma_trans(l2_out, ddr_out, out_dma_size);

            ddr_in += elem_per_iter;
            ddr_out += elem_per_iter;
            elem_left -= elem_per_iter;
        }
    } 
    // Case 3: 广播模式，因随机访存且位置不连续，不使用L2，直接在DDR中读写计算
    else {
        int total_elements = get_total_elements(num_dims, output_dims);

        // 由0号逻辑核更新共享空间的strides，避免多核并发写冲突
        if (logic_core_id == 0) {
            strides0[num_dims - 1] = 1;
            strides1[num_dims - 1] = 1;
            strides_output[num_dims - 1] = 1;
            for (i = num_dims - 2; i >= 0; --i) {
                strides0[i] = strides0[i + 1] * input0_dims[i + 1];
                strides1[i] = strides1[i + 1] * input1_dims[i + 1];
                strides_output[i] = strides_output[i + 1] * output_dims[i + 1];
            }
        }
        
        // 核心同步，等待0号核配置完毕
        C6678E_SyncN(core_num, logic_core_id);

        int elem_per_core = (total_elements + core_num - 1) / core_num;
        int start_index = logic_core_id * elem_per_core;
        int end_index = min(start_index + elem_per_core, total_elements);
        int elem_this_core = end_index - start_index;

        if (elem_this_core > 0) {
            for (i = start_index; i < end_index; ++i) {
                int offset0 = 0;
                int offset1 = 0;
                int temp_i = i;
                int d;
                for (d = 0; d < num_dims; ++d) {
                    int coord = temp_i / strides_output[d];
                    temp_i %= strides_output[d];

                    if (input0_dims[d] > 1) {
                        offset0 += coord * strides0[d];
                    }
                    if (input1_dims[d] > 1) {
                        offset1 += coord * strides1[d];
                    }
                }
                output[i] = input0[offset0] >= input1[offset1];
            }
        }
    }
}
```
