#include <tk/tkernel.h>
#include <tm/tmonitor.h>
#include <stdbool.h>

extern bool ai_init(void);
extern int ai_run_inference(void);

ID tskid_sensor, tskid_ai, tskid_control, tskid_ui;
ID semid_ai;
int simulated_moisture = 40;

/* Priority 5: Sensor Acquisition Task */
void SensorTask(INT stacd, void *exinf) {
    while(1) {
        simulated_moisture = (simulated_moisture + 5) % 100;
        tm_printf("SensorTask: Moisture simulated at %d%%\n", simulated_moisture);

        tk_sig_sem(semid_ai, 1);
        tk_dly_tsk(3000);
    }
}

/* Priority 7: AI Inference Task */
void AITask(INT stacd, void *exinf) {
    ai_init();
    while(1) {
        tk_wai_sem(semid_ai, 1, TMO_FEVR);
        tm_printf("AITask: Running Leaf Disease Inference on Ethos-U55...\n");

        int result = ai_run_inference();
        tm_printf("AITask: Disease Class Detected: %d\n", result);

        tk_wup_tsk(tskid_control);
    }
}

/* Priority 8: Actuator Control Task */
void ControlTask(INT stacd, void *exinf) {
    while(1) {
        tk_slp_tsk(TMO_FEVR);
        tm_printf("ControlTask: Checking AI results. Simulated Valve Actuated.\n");
        tk_wup_tsk(tskid_ui);
    }
}

/* Priority 10: UI & Logging Task */
void UITask(INT stacd, void *exinf) {
    while(1) {
        tk_slp_tsk(TMO_FEVR);
        tm_printf("UITask: Logging data. (Ready for Display Update)\n\n");
    }
}

EXPORT INT usermain(void) {
    T_CSEM csem = { .sematr = TA_TFIFO, .isemcnt = 0, .maxsem = 1 };
    semid_ai = tk_cre_sem(&csem);

    T_CTSK ctsk_sensor  = { .task = SensorTask,  .itskpri = 5,  .stksz = 1024 };
    T_CTSK ctsk_ai      = { .task = AITask,      .itskpri = 7,  .stksz = 8192 };
    T_CTSK ctsk_control = { .task = ControlTask, .itskpri = 8,  .stksz = 1024 };
    T_CTSK ctsk_ui      = { .task = UITask,      .itskpri = 10, .stksz = 1024 };

    tskid_sensor  = tk_cre_tsk(&ctsk_sensor);
    tskid_ai      = tk_cre_tsk(&ctsk_ai);
    tskid_control = tk_cre_tsk(&ctsk_control);
    tskid_ui      = tk_cre_tsk(&ctsk_ui);

    tk_sta_tsk(tskid_sensor, 0);
    tk_sta_tsk(tskid_ai, 0);
    tk_sta_tsk(tskid_control, 0);
    tk_sta_tsk(tskid_ui, 0);

    tk_slp_tsk(TMO_FEVR);
    return 0;
}
