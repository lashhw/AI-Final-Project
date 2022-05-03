<script>
import LineChart from './components/LineChart.vue'

export default {
  data() {
    return {
      id_input: null,
      id: null,
      prediction: {},
      history: {},
      chart_options: {
        scales: {
            x: {
                type: 'time'
            }
        }
      }
    }
  },
  computed: {
    chart_prediction_data() {
      return {
        labels: Object.keys(this.prediction),
        datasets: [{
            label: this.id,
            data: Object.values(this.prediction) 
        }]
      }
    },
    chart_history_data() {
      return {
        labels: Object.keys(this.history),
        datasets: [{
            label: this.id,
            data: Object.values(this.history) 
        }]
      }
    }
  },
  methods: {
    get_data() {
      fetch(
        `https://ntpcparking.azurewebsites.net/api/gethistory?id=${this.id_input}&max_days=0`
      ).then(response => {
        return response.json()
      }).then(result => {
        this.history = result
      })

      fetch(
        `https://ntpcparking.azurewebsites.net/api/predictfuture?id=${this.id_input}`
      ).then(response => {
        return response.json()
      }).then(result => {
        this.prediction = result
      })

      this.id = this.id_input
    }
  },
  components: { LineChart }
}
</script>

<template>
  <input v-model="id_input" placeholder="停車場 ID" />
  <button @click.prevent="get_data"> 送出 </button>
  <LineChart :chart-data='chart_history_data' :chart-options="chart_options" />
  <LineChart :chart-data='chart_prediction_data' :chart-options="chart_options" />
</template>

<style>

</style>
