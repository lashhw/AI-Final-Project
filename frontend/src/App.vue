<script>
import AreaChart from './components/AreaChart.vue'
import info_json from './assets/info.json'

export default {
  data() {
    return {
      info: info_json,
      selection: {
        area: "",
        id: ""
      },
      history: {},
      prediction: {},
      chart_options: {
        xaxis: {
          type: "datetime"
        },
        yaxis: {
          decimalsInFloat: 0
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
    chart_series() {
      return [
        {
          name: "history",
          data: Object.entries(this.history)
        },
        {
          name: "prediction",
          data: Object.entries(this.prediction)
        }
      ]
    },
    area_list() {
      return [...new Set(this.info.map(item => item.AREA))]
    }
  },
  methods: {
    async get_data() {
      const res_history = await fetch(`https://ntpcparking.azurewebsites.net/api/gethistory?id=${this.selection.id}&max_days=0`)
      this.history = await res_history.json()

      const res_prediction = await fetch(`https://ntpcparking.azurewebsites.net/api/predictfuture?id=${this.selection.id}`)
      this.prediction = await res_prediction.json()
    }
  },
  components: { AreaChart }
}
</script>

<template>
  <select v-model="selection.area">
    <option value="" disabled>請選擇地區</option>
    <option v-for="area in area_list">
      {{ area }}
    </option>
  </select>
  <select v-model="selection.id" @change="get_data">
    <option value="" disabled>請選擇停車場名稱</option>
    <template v-for="x in info">
      <option v-if="x.AREA === selection.area" :value="x.ID">
        {{ x.NAME }}
      </option>
    </template>
  </select>
  <AreaChart :series="chart_series" :options="chart_options" />
</template>

<style>

</style>
