<template>
  <q-page padding>
    <div class="q-gutter-md">
      <div class="row flex-center q-gutter-sm">
        <div class="col-5">
          <q-select v-model="selection.area" :options="area_list" label="行政區" />
        </div>
        <div class="col-5">
          <q-select v-model="selection.lot" :options="p_lot_list" label="停車場名稱" />
        </div>
      </div>
      <div class="row flex-center q-gutter-sm">
        <q-checkbox v-model="enable_baseline" label="顯示 baseline" />
        <q-icon v-show="warning" name="warning" color="negative" size="2em" />
        <q-icon v-show="done" name="done" color="primary" size="2em" />
        <q-spinner-dots v-show="loading" color="primary" size="2em" />
      </div>
      <div class="row flex-center">
        <div class="col-12 col-xs-10 col-sm-8 col-md-4">
          <AreaChart :series="chart_series" :options="chart_options" />
        </div>
        <div class="col-12 col-xs-10 col-sm-8 col-md-4">
          <AreaChart :series="chart_weekly_series" :options="chart_weekly_options" />
        </div>
      </div>
    </div>
  </q-page>
</template>

<script>
import AreaChart from '../components/AreaChart.vue'
import info_json from '../assets/info.json'

export default {
  name: 'IndexPage',
  data() {
    return {
      info: info_json,
      selection: {
        area: "",
        lot: ""
      },
      state: "done",
      enable_baseline: false,
      chart_series_all: [],
      chart_weekly_series_all: [],
      chart_options: {
        title : {
          text: "一天內車位數趨勢"
        },
        xaxis: {
          type: "datetime",
          labels: {
            datetimeUTC: false
          }
        },
        yaxis: {
          decimalsInFloat: 0
        },
        stroke: {
          dashArray: [0, 4, 4, 4]
        },
        dataLabels: {
          enabled: false
        },
        tooltip: {
          x: {
            format: "dd MMM hh:mm TT"
          }
        },
        annotations: {
          xaxis: [
            {
              x: Date.now(),
              label: {
                text: "now"
              }
            }
          ]
        }
      },
      chart_weekly_options: {
        title : {
          text: "一週內車位數趨勢"
        },
        colors: ['#FEB019', '#FF4560'],
        xaxis: {
          type: "datetime",
          labels: {
            datetimeUTC: false
          }
        },
        yaxis: {
          decimalsInFloat: 0
        },
        stroke: {
          dashArray: [4, 4]
        },
        dataLabels: {
          enabled: false
        },
        tooltip: {
          x: {
            format: "dd MMM hh:mm TT"
          }
        }
      }
    }
  },
  computed: {
    area_list() {
      return [...new Set(this.info.map(item => item.AREA))]
    },
    p_lot_list() {
      return this.info.filter(x => x.AREA === this.selection.area)
                      .map(x => x.NAME)
    },
    chart_series() {
      if (this.enable_baseline) 
        return this.chart_series_all
      else
        return this.chart_series_all.slice(0, -1)
    },
    chart_weekly_series() {
      if (this.enable_baseline) 
        return this.chart_weekly_series_all
      else
        return this.chart_weekly_series_all.slice(0, -1)
    },
    loading() {
      return this.state == 'loading'
    },
    warning() {
      return this.state == 'warning'
    },
    done() {
      return this.state == 'done'
    }
  },
  watch: {
    'selection.lot'() {
      for (var i = 0; i < this.info.length; i++) {
        if (this.info[i].NAME === this.selection.lot) {
          this.update_series(this.info[i].ID)
          return
        }
      }
    }
  },
  methods: {
    async update_series(id) {
      this.state = 'loading';
      try {
        const res_history = await fetch(`https://ntpcparking.azurewebsites.net/api/gethistory?id=${id}`)
        var history = await res_history.json()

        const res_prediction = await fetch(`https://ntpcparking.azurewebsites.net/api/predictfuture?id=${id}`)
        var prediction = await res_prediction.json()

        var history_pairs = Object.entries(history)
        for (var i = 0; i < history_pairs.length; i++) {
          var splitted = history_pairs[i][0].split(/[- :]/)
          history_pairs[i][0] = Date.UTC(splitted[0], splitted[1]-1, splitted[2], splitted[3], splitted[4])
        }
        var now = Date.now()
        history_pairs = history_pairs.filter(pair => pair[0] >= now - 43200000)

        var lstm_pred_pairs = Object.entries(prediction['lstm'])
        for (var i = 0; i < lstm_pred_pairs.length; i++) {
          var splitted = lstm_pred_pairs[i][0].split(/[- :]/)
          lstm_pred_pairs[i][0] = Date.UTC(splitted[0], splitted[1]-1, splitted[2], splitted[3], splitted[4])
        }
        var last_history_time = history_pairs[history_pairs.length-1][0]
        lstm_pred_pairs = lstm_pred_pairs.filter(pair => pair[0] > last_history_time)

        var prophet_pred_pairs = Object.entries(prediction['prophet_yhat'])
        for (var i = 0; i < prophet_pred_pairs.length; i++) {
          var splitted = prophet_pred_pairs[i][0].split(/[- :]/)
          prophet_pred_pairs[i][0] = Date.UTC(splitted[0], splitted[1]-1, splitted[2], splitted[3], splitted[4])
        }

        var average_pred_pairs = Object.entries(prediction['average'])
        for (var i = 0; i < average_pred_pairs.length; i++) {
          var splitted = average_pred_pairs[i][0].split(/[- :]/)
          average_pred_pairs[i][0] = Date.UTC(splitted[0], splitted[1]-1, splitted[2], splitted[3], splitted[4])
        }
      
        this.chart_series_all = [
          {
            name: 'History',
            data: history_pairs
          },
          {
            name: 'LSTM',
            data: lstm_pred_pairs
          },
          {
            name: 'Prophet',
            data: prophet_pred_pairs.slice(0, 96)
          },
          {
            name: 'Average', 
            data: average_pred_pairs.slice(0, 96)
          }
        ]

        this.chart_weekly_series_all = [
          {
            name: 'Prophet',
            data: prophet_pred_pairs
          },
          { 
            name: 'Average', 
            data: average_pred_pairs 
          }
        ]

        this.state = 'done'
      } catch (error) {
        this.chart_series_all = []
        this.chart_weekly_series_all = []
        this.state = 'warning'
      }
    }
  },
  components: { AreaChart }
}
</script>